import sys, os

# Add parent directory to path (for 'models' package)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Add current directory to path (for 'routing_core' sibling module)
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS 
from routing_core import get_routes_and_metrics, _build_or_load_graph
from routing_core import snap_to_nearest_node, get_pollution_points
import logging

# --- CONFIGURATION ---
# Set up Flask app
app = Flask(__name__, static_folder='../frontend')
CORS(app)  # Enable CORS for local development (frontend on one port, backend on another)

# Configure logging to see errors
logging.basicConfig(level=logging.INFO)

# --- ROUTING ENDPOINTS ---

@app.route('/')
def serve_index():
    """Serves the main HTML file for the application."""
    try:
        # Assumes index.html is in the ../frontend/ directory
        return send_from_directory(app.static_folder, 'index.html')
    except Exception as e:
        return f"Error serving index.html: {e}", 500

@app.route('/<path:path>')
def serve_static_files(path):
    """Serves other static files (script.js, style.css) from the frontend directory."""
    return send_from_directory(app.static_folder, path)

@app.route('/api/snap', methods=['POST'])
def snap():
    """
    POST JSON: { "lat": 28.61, "lon": 77.20 }
    Returns: { "lat": 28.61, "lon": 77.20, "node": 12345 }
    """
    try:
        payload = request.get_json(force=True)
        lat = float(payload.get('lat'))
        lon = float(payload.get('lon'))
    except Exception as e:
        return jsonify({'error': f'Invalid payload: {e}'}), 400

    try:
        snapped = snap_to_nearest_node((lat, lon))
        return jsonify(snapped), 200
    except Exception as e:
        return jsonify({'error': f'Failed to snap point: {e}'}), 500

@app.route('/api/pollution_points', methods=['GET'])
def pollution_points():
    """
    Returns predicted station points for heatmap. Response:
    [{ "lat": 28.6, "lon": 77.2, "ps": 320 }, ...]
    """
    try:
        pts = get_pollution_points()
        return jsonify(pts), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/route', methods=['POST'])
def route():
    try:
        data = request.get_json(force=True) or {}

        # Require start and end to be provided by the user.
        start_str = data.get('start')
        end_str = data.get('end')
        if not start_str or not end_str:
            return jsonify({
                'error': 'Missing start or end coordinates. Please provide "start" and "end" fields in "lat,lon" format.'
            }), 400

        # Parse weight (optional; default to 0.5 if not provided but explicit requirement won't use defaults for places)
        try:
            user_weight = float(data.get('weight', 0.5))
            if not (0.0 <= user_weight <= 1.0):
                raise ValueError("weight must be between 0 and 1")
        except Exception as ex:
            return jsonify({'error': f'Invalid weight value: {ex}'}), 400

        # Convert strings "lat, lon" to tuple (lat, lon)
        try:
            start_coords = tuple(map(float, [s.strip() for s in start_str.split(',')]))
            end_coords = tuple(map(float, [s.strip() for s in end_str.split(',')]))
        except Exception:
            return jsonify({'error': 'Invalid coordinate format. Must be "lat, lon" with numeric values.'}), 400

        if len(start_coords) != 2 or len(end_coords) != 2:
            return jsonify({'error': 'Invalid coordinate format. Must be "lat, lon".'}), 400

        logging.info(f"Finding routes from {start_coords} to {end_coords} with W={user_weight}")

        # Call core logic
        route_results = get_routes_and_metrics(start_coords, end_coords, user_weight)

        # If the core returns an error-like dict, pass that as a 400
        if isinstance(route_results, dict) and route_results.get('error'):
            return jsonify(route_results), 400

        return jsonify(route_results), 200

    except Exception as e:
        logging.exception("Error during routing process")
        return jsonify({
            'error': 'An internal error occurred during route calculation.',
            'detail': str(e)
        }), 500


@app.route('/api/traffic_refresh', methods=['POST'])
def traffic_refresh():

    try:
        payload = request.get_json(silent=True) or {}
        points = payload.get('points', None)
        max_points = int(payload.get('max_points', 200))
        spacing_deg = float(payload.get('spacing_deg', 0.005))

        import tomtom_integration as tt
        res = tt.update_graph_from_tomtom_points(points=points, max_points=max_points, spacing_deg=spacing_deg)
        return jsonify({'ok': True, 'result': res}), 200
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/traffic_status', methods=['GET'])
def traffic_status():
    # returns a trivial confirmation
    return jsonify({'tomtom_key_present': bool(os.environ.get('TOMTOM_API_KEY'))}), 200

# --- RUNNING THE APPLICATION ---
if __name__ == '__main__':
    # Ensure the data directory exists
    if not os.path.exists('./data'):
        os.makedirs('./data')
        print("Created './data' directory. Place ERA5 parquet file here if you have one.")
    
    print("\n--- FLASK SERVER STARTED ---")
    print(f"Access the application at http://127.0.0.1:5000/")
    app.run(debug=True, port=5000, use_reloader=False)
