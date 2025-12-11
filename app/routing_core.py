import os
import time
import requests
import networkx as nx
import pandas as pd
import numpy as np
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial import cKDTree
from typing import Dict, List, Tuple, Any

from models.model_loader import run_model_prediction  # your AQI model

PLACE = "New Delhi, India"

G_proj = None
G_orig = None
PLACE_POLYGON_GDF = None

# ==============================
# ✅ EMISSION MODEL (INDIAN NORMS)
# ==============================
def emission_factor_from_indian_norms(fuel_type: str, bharat_stage: str) -> float:
    """
    Conceptual mapping from Indian emission standards (Bharat Stage) + fuel type
    to a normalized emission factor (0 = clean, 1 = very polluting).
    """
    fuel_type = (fuel_type or "").lower()
    bharat_stage = (bharat_stage or "").upper()

    # Base factor by fuel type
    if fuel_type in ["ev", "electric"]:
        base = 0.0
    elif fuel_type == "cng":
        base = 0.3
    elif fuel_type == "petrol":
        base = 0.6
    elif fuel_type == "diesel":
        base = 0.8
    else:
        base = 0.5  # unknown / mix

    # Adjustment by Bharat Stage (older norms = dirtier)
    stage_adjust = {
        "BS1": 1.0,
        "BS2": 0.9,
        "BS3": 0.8,
        "BS4": 0.6,
        "BS5": 0.4,
        "BS6": 0.2,
    }

    adj = stage_adjust.get(bharat_stage, 0.6)
    ef = (base + adj) / 2.0  # simple average → 0–1

    return max(0.0, min(1.0, ef))


# ==============================
# ✅ 3-VARIABLE MATRIX COST (T, P, E)
# ==============================
def compute_green_cost(T: float, P: float, E: float,
                       w_T: float = 0.3,
                       w_P: float = 0.4,
                       w_E: float = 0.3) -> float:
    """
    3-variable matrix-based cost for one edge:
      x = [T, P, E]^T
      w = [w_T, w_P, w_E]^T
      C = w^T x

    T = normalized travel time (0–1)
    P = normalized pollution score (0–1)
    E = normalized emission factor (0–1)
    """
    T = max(0.0, min(1.0, T))
    P = max(0.0, min(1.0, P))
    E = max(0.0, min(1.0, E))
    return w_T * T + w_P * P + w_E * E


# ==============================
# ✅ LIVE AQI FORECAST → POLLUTION SCORE
# ==============================
def get_forecast_data_from_model() -> pd.DataFrame:
    """
    Fetch live meteorological data, run your AQI model, and
    create pseudo-stations around Delhi with a Pollution Score PS.
    """
    try:
        lat, lon = 28.6139, 77.2090

        url = (
            "https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            "&hourly=temperature_2m,wind_speed_10m,relative_humidity_2m,surface_pressure"
            "&forecast_days=3"
        )

        r = requests.get(url, timeout=10)
        r.raise_for_status()
        hourly = r.json()["hourly"]

        df = pd.DataFrame({
            "temperature": hourly["temperature_2m"],
            "wind_speed": hourly["wind_speed_10m"],
            "humidity": hourly["relative_humidity_2m"],
            "pressure": hourly["surface_pressure"],
        })

        # Your ML model gives predicted AQI
        predicted_aqi = run_model_prediction(df)

        # For simplicity, use AQI directly as pollution score
        df["PS"] = predicted_aqi

        # Scatter virtual stations around Delhi center
        center = (28.6129, 77.2295)
        n = len(df)
        df["station_lat"] = center[0] + (np.random.rand(n) - 0.5) * 0.1
        df["station_lon"] = center[1] + (np.random.rand(n) - 0.5) * 0.1

        return df[["station_lat", "station_lon", "PS"]]

    except Exception:
        # Fallback if API/ML fails so app still works
        return pd.DataFrame({
            "station_lat": [28.635, 28.535, 28.700, 28.580],
            "station_lon": [77.225, 77.200, 77.100, 77.300],
            "PS": [400, 180, 550, 250],
        })


# ==============================
# ✅ GRAPH LOADING
# ==============================
def _build_or_load_graph():
    global G_proj, G_orig

    if G_proj is None or G_orig is None:
        print("Loading road network...")
        # Use a smaller radius (2000m) to stay within Render's free tier memory limits (512MB).
        # "New Delhi" as a place is too large and causes OOM kills.
        G_orig = ox.graph_from_point((28.6139, 77.2090), dist=2000, network_type="drive")
        G_proj = ox.project_graph(G_orig)

        # Basic travel time estimate (length / speed)
        for u, v, k, data in G_proj.edges(keys=True, data=True):
            length = data.get("length", 100)  # meters
            # Approx 30 km/h → 8.33 m/s
            data["travel_time"] = length / 8.33

    return G_proj


# ==============================
# ✅ POLLUTION ASSIGNMENT (IDW)
# ==============================
def _assign_pollution_score_and_norms(G_proj, aqi_data):
    """
    Assign Pollution_Score to each edge via IDW interpolation.
    Also compute normalized time (T) and pollution (P) for matrix routing.
    """
    edges = ox.graph_to_gdfs(G_proj, nodes=False, edges=True).reset_index()

    stations = aqi_data.copy()
    stations["geometry"] = [Point(xy) for xy in zip(stations["station_lon"], stations["station_lat"])]
    stations_gdf = gpd.GeoDataFrame(stations, geometry="geometry", crs="EPSG:4326")

    stations_proj = stations_gdf.to_crs(edges.crs)

    tree = cKDTree([(p.x, p.y) for p in stations_proj.geometry])
    ps_vals = stations_proj["PS"].to_numpy()

    # --- 1) Assign Pollution_Score ---
    for idx, row in edges.iterrows():
        pt = row.geometry.centroid
        dist, ind = tree.query([pt.x, pt.y])
        G_proj.edges[row.u, row.v, row.key]["Pollution_Score"] = float(ps_vals[ind])

    # --- 2) Compute normalized time & pollution for all edges ---
    times = []
    polls = []
    edge_keys = []

    for u, v, k, data in G_proj.edges(keys=True, data=True):
        t_val = float(data.get("travel_time", 60.0))
        p_val = float(data.get("Pollution_Score", 300.0))
        times.append(t_val)
        polls.append(p_val)
        edge_keys.append((u, v, k))

    times = np.array(times)
    polls = np.array(polls)

    t_min, t_max = times.min(), times.max()
    p_min, p_max = polls.min(), polls.max()

    t_range = (t_max - t_min) if t_max > t_min else 1.0
    p_range = (p_max - p_min) if p_max > p_min else 1.0

    for (u, v, k), t_val, p_val in zip(edge_keys, times, polls):
        G_proj.edges[u, v, k]["time_norm"] = float((t_val - t_min) / t_range)
        G_proj.edges[u, v, k]["poll_norm"] = float((p_val - p_min) / p_range)

    return G_proj


# ==============================
# ✅ ROUTING MAIN FUNCTION (3-VARIABLE MATRIX)
# ==============================
def get_routes_and_metrics(start_coords, end_coords, user_weight):
    """
    Compute:
      - main_route: using 3-variable matrix cost (time, pollution, emission)
      - fastest_route: using only travel time

    user_weight can be used to slightly bias pollution vs time if needed.
    """
    aqi_data = get_forecast_data_from_model()
    G_proj = _build_or_load_graph()
    G_proj = _assign_pollution_score_and_norms(G_proj, aqi_data)

    # Nearest nodes in original graph (lat, lon)
    orig = ox.nearest_nodes(G_orig, start_coords[1], start_coords[0])
    dest = ox.nearest_nodes(G_orig, end_coords[1], end_coords[0])

    # For now, use a conceptual average emission factor E (e.g. mixed BS-IV/BS-VI fleet)
    # In future, this can be made dynamic from real vehicle data.
    avg_emission_factor = emission_factor_from_indian_norms("diesel", "BS4")

    # Optionally adjust weights using user_weight (0–1) to emphasize pollution vs time
    # Example: more weight on pollution when user_weight is high
    base_w_T, base_w_P, base_w_E = 0.3, 0.4, 0.3
    w_T = base_w_T * (1 - 0.3 * user_weight)
    w_P = base_w_P * (1 + 0.3 * user_weight)
    w_E = base_w_E

    def matrix_cost(u, v, d):
        T = d.get("time_norm", 0.5)
        P = d.get("poll_norm", 0.5)
        E = avg_emission_factor
        return compute_green_cost(T, P, E, w_T=w_T, w_P=w_P, w_E=w_E)

    # --- Main (green) route using matrix cost ---
    main_route = nx.shortest_path(
        G_proj,
        orig,
        dest,
        weight=matrix_cost
    )

    # --- Fastest route using raw travel_time only ---
    fast_route = nx.shortest_path(
        G_proj,
        orig,
        dest,
        weight="travel_time"
    )

    def get_coords(route):
        return [(G_orig.nodes[n]["y"], G_orig.nodes[n]["x"]) for n in route]

    # Simple metrics for now (you can extend with real time/PS sums)
    metrics = {
        "Main_Route": {
            "Time_min": len(main_route),  # placeholder: hop count
        },
        "Fastest_Route": {
            "Time_min": len(fast_route),
        }
    }

    return {
        "main_route_coords": get_coords(main_route),
        "fastest_route_coords": get_coords(fast_route),
        "metrics": metrics,
        "map_center": [start_coords[0], start_coords[1]],
    }


# ==============================
# ✅ FRONTEND HELPERS
# ==============================
def snap_to_nearest_node(latlon: Tuple[float, float]) -> Dict[str, Any]:
    """
    Given (lat, lon) returns snapped node and its coordinates.
    """
    _build_or_load_graph()
    lon = float(latlon[1])
    lat = float(latlon[0])

    node = ox.nearest_nodes(G_orig, lon, lat)
    return {
        "lat": float(G_orig.nodes[node]["y"]),
        "lon": float(G_orig.nodes[node]["x"]),
        "node": int(node),
    }



# ==============================
# ✅ SPATIAL INDEX HELPER
# ==============================
def _edge_midpoints_and_index(G):
    """
    Returns (coords, idx_map) for building a cKDTree of edge midpoints.
    coords: (M, 2) array of (lat, lon)
    idx_map: dict { index_in_coords -> (u, v, k) }
    """
    coords = []
    idx_map = {}
    
    if G is None:
        return np.array([]), {}

    idx = 0
    for u, v, k, data in G.edges(keys=True, data=True):
        # We try to use 'geometry' if present (accurate midpoint)
        # otherwise average of node coords.
        try:
            if "geometry" in data:
                mid = data["geometry"].interpolate(0.5, normalized=True)
                # If these are shapely points, accessing .y / .x works
                coords.append([mid.y, mid.x])
            else:
                n1 = G.nodes[u]
                n2 = G.nodes[v]
                lat = (n1['y'] + n2['y']) / 2
                lon = (n1['x'] + n2['x']) / 2
                coords.append([lat, lon])
            
            idx_map[idx] = (u, v, k)
            idx += 1
        except Exception:
            continue

    return np.array(coords), idx_map


_POLLUTION_CACHE = None
_POLLUTION_CACHE_TS = 0.0
_POLLUTION_CACHE_TTL = 60.0  # seconds


def get_pollution_points() -> List[Dict[str, Any]]:
    """
    Returns list of station points for heatmap:
      [{ "lat": ..., "lon": ..., "ps": ... }, ...]
    """
    global _POLLUTION_CACHE, _POLLUTION_CACHE_TS

    now = time.time()
    if _POLLUTION_CACHE and (now - _POLLUTION_CACHE_TS) < _POLLUTION_CACHE_TTL:
        return _POLLUTION_CACHE

    df = get_forecast_data_from_model()
    out = []

    for _, row in df.iterrows():
        out.append({
            "lat": float(row["station_lat"]),
            "lon": float(row["station_lon"]),
            "ps": float(row["PS"]),
        })

    _POLLUTION_CACHE = out
    _POLLUTION_CACHE_TS = now
    return out
