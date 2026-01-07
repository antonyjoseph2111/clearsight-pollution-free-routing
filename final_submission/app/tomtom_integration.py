# app/tomtom_integration.py
import os
import time
import math
import requests
import numpy as np
from typing import List, Dict, Tuple, Any

# TomTom Flow Segment point endpoint (we query single points)
TOMTOM_KEY = os.environ.get('TOMTOM_API_KEY', None)
TOMTOM_FLOW_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"

# Safety: do not hammer API. Use friendly defaults.
DEFAULT_MAX_POINTS = 200
DEFAULT_POINT_SPACING_DEG = 0.005  # ~500m (approx). Increase for sparser sampling.
REQUEST_TIMEOUT = 6.0

def query_tomtom_flow_point(lat: float, lon: float) -> Dict[str, Any]:
    """
    Query TomTom flowSegmentData for a single lat/lon.
    Returns { currentSpeed, freeFlowSpeed, confidence } or None on failure.
    """
    if not TOMTOM_KEY:
        # Running without key -> skip
        return None
    params = {
        "point": f"{lat},{lon}",
        "unit": "KMPH",
        "key": TOMTOM_KEY
    }
    try:
        r = requests.get(TOMTOM_FLOW_URL, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        payload = r.json()
        fs = payload.get('flowSegmentData') or {}
        return {
            'currentSpeed': fs.get('currentSpeed'),
            'freeFlowSpeed': fs.get('freeFlowSpeed'),
            'confidence': fs.get('confidence')
        }
    except Exception as e:
        # Log to console; real app should use logger
        print(f"[tomtom_integration] query failed for {lat},{lon}: {e} (status={getattr(r,'status_code',None)})")
        return None

# Helper to build grid sample over graph bounding box
def _sample_points_over_bounds(bounds: Tuple[float,float,float,float], spacing_deg: float, max_points: int):
    """
    bounds = (min_lat, min_lon, max_lat, max_lon)
    Return list of (lat, lon) sample points (limited to max_points).
    """
    min_lat, min_lon, max_lat, max_lon = bounds
    lat_vals = np.arange(min_lat, max_lat + 1e-9, spacing_deg)
    lon_vals = np.arange(min_lon, max_lon + 1e-9, spacing_deg)
    pts = []
    for la in lat_vals:
        for lo in lon_vals:
            pts.append((float(la), float(lo)))
            if len(pts) >= max_points:
                return pts
    return pts

def update_graph_from_tomtom_points(points: List[Dict[str, float]],
                                    search_k: int = 1,
                                    max_points: int = DEFAULT_MAX_POINTS,
                                    spacing_deg: float = DEFAULT_POINT_SPACING_DEG) -> Dict[str, Any]:

    # lazy import to avoid circular imports
    from routing_core import _edge_midpoints_and_index, G_proj, G_orig, _build_or_load_graph
    if G_proj is None:
        _build_or_load_graph()

    # Build search graph (use original geometry if available)
    search_graph = G_orig if (G_orig is not None) else G_proj

    # If no explicit points provided, sample grid over graph bounds
    if not points:
        # get bbox of nodes (lat, lon are stored as y,x)
        ys = [data['y'] for _, data in search_graph.nodes(data=True)]
        xs = [data['x'] for _, data in search_graph.nodes(data=True)]
        min_lat, max_lat = min(ys), max(ys)
        min_lon, max_lon = min(xs), max(xs)
        grid_pts = _sample_points_over_bounds((min_lat, min_lon, max_lat, max_lon),
                                             spacing_deg=spacing_deg,
                                             max_points=max_points)
        points_to_query = [{'lat': float(p[0]), 'lon': float(p[1])} for p in grid_pts]
    else:
        points_to_query = [{'lat': float(p['lat']), 'lon': float(p['lon'])} for p in points][:max_points]

    # Build KD-tree over edge midpoints and index map
    coords, idx_map = _edge_midpoints_and_index(search_graph)
    if coords.size == 0 or len(idx_map) == 0:
        return {'updated_edges': 0, 'queried_points': 0, 'timestamp': time.time(), 'note':'no edges in graph'}
    from scipy.spatial import cKDTree
    tree = cKDTree(coords)

    updates = {}  # mapping (u,v,k) -> observed_speed_kph
    queried = 0
    for p in points_to_query:
        lat, lon = p['lat'], p['lon']
        res = query_tomtom_flow_point(lat, lon)
        queried += 1
        if not res or res.get('currentSpeed') is None:
            continue
        time.sleep(1.5)

        speed_kph = float(res['currentSpeed'])
        # nearest edge(s)
        dist, idx = tree.query([lat, lon], k=search_k)
        if np.isscalar(idx):
            idxs = [int(idx)]
        else:
            idxs = [int(i) for i in idx]
        for i in idxs:
            u, v, k = idx_map[i]
            updates[(u, v, k)] = speed_kph

    # Apply updates to projected graph (G_proj)
    updated_edges = 0
    for (u, v, k), obs_speed_kph in updates.items():
        try:
            data = G_proj.edges[u, v, k]
            length_m = data.get('length', None)
            if length_m is None or obs_speed_kph <= 0:
                continue
            # compute new travel time (seconds)
            travel_time_s = float(length_m) / (obs_speed_kph / 3.6)
            G_proj.edges[u, v, k]['travel_time'] = travel_time_s
            G_proj.edges[u, v, k]['observed_speed_kph'] = obs_speed_kph
            updated_edges += 1
        except Exception:
            # some edges may mismatch between search_graph and G_proj keying; skip gracefully
            continue

    # Recompute normalized time_norm and poll_norm across the graph
    times = []
    ps_list = []
    for _, _, _, data in G_proj.edges(keys=True, data=True):
        times.append(data.get('travel_time', 300.0))
        ps_list.append(data.get('Pollution_Score', 500.0))
    if len(times) == 0:
        return {'updated_edges': updated_edges, 'queried_points': queried, 'timestamp': time.time()}

    times = np.array(times, dtype=float)
    ps_list = np.array(ps_list, dtype=float)
    t_min, t_max = times.min(), times.max()
    p_min, p_max = ps_list.min(), ps_list.max()
    t_range = t_max - t_min if t_max > t_min else 1.0
    p_range = p_max - p_min if p_max > p_min else 1.0

    # iterate again to set norms (use same ordering as edges())
    for (u, v, k), t, p in zip(list(G_proj.edges(keys=True)), times, ps_list):
        try:
            G_proj.edges[u, v, k]['time_norm'] = float((t - t_min) / t_range)
            G_proj.edges[u, v, k]['poll_norm'] = float((p - p_min) / p_range)
        except Exception:
            continue

    return {'updated_edges': updated_edges, 'queried_points': queried, 'timestamp': time.time()}
