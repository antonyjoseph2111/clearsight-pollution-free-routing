# 🌬️ ClearSight - Pollution-Free Routing System

**SIH 2024 Project - Auralis**

ClearSight is an advanced AI-powered routing application designed to help commuters in Delhi-NCR find the healthiest travel routes. By integrating live AQI (Air Quality Index) forecasts with road network data, it calculates paths that minimize exposure to harmful pollutants (PM2.5, PM10) while maintaining reasonable travel times.

---

## 🚀 Key Features

*   **Pollution-Aware Navigation**: unique routing algorithm that factors in both *Travel Time* and *Pollution Exposure*.
*   **Live AQI Forecasting**: Uses real-time meteorological data (Wind, Temp, Humidity) and an LSTM Deep Learning model to predict pollution hotspots.
*   **3-Variable Cost Matrix**: Optimizes for Time, Pollution Score, and Emission Factors (Vehicle Standard).
*   **Interactive Heatmap**: Visualizes predicted pollution intensity across the city.
*   **User Preference Control**: Adjust the priority between "Fastest" and "Cleanest" routes using a simple slider.
*   **Live Traffic Integration**: (Optional) Can fetch real-time traffic data via TomTom API.

## 🛠️ Technology Stack

*   **Frontend**: HTML5, Tailwind CSS, Leaflet.js (for maps).
*   **Backend**: Python, Flask, NetworkX (Graph Logic), OSMnx (OpenStreetMap Data).
*   **AI/ML**: TensorFlow/Keras (LSTM Model), Scikit-Learn.
*   **Data Source**: Open-Meteo API (Weather), OpenStreetMap (Roads).
*   **Hosting**: Render (Web Service).

---

## ⚙️ How It Works (The "Zero Intervention" Deployment)

This project is configured for **Continuous Deployment** on Render.
1.  **Code Updates**: Any commit pushed to the `main` branch on GitHub automatically triggers a new build on Render.
2.  **Serverless Build**: Render installs dependencies from `requirements.txt` and starts the Gunicorn server.
3.  **Live**: The app serves the React-like frontend and API endpoints immediately.

### Local Installation

If you want to run it on your own machine:

1.  **Clone the Repo**:
    ```bash
    git clone https://github.com/antonyjoseph2111/clearsight-pollution-free-routing.git
    cd clearsight-pollution-free-routing
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Server**:
    ```bash
    python app/app.py
    ```

4.  **Access**: Open `http://127.0.0.1:5000` in your browser.

---

## 📡 API Endpoints

The backend provides a REST API for the frontend:

*   `GET /api/pollution_points`: Returns predicted AQI for heatmap visualization.
*   `POST /api/route`: Calculates optimal path.
    *   **Body**: `{ "start": "lat,lon", "end": "lat,lon", "weight": 0.5 }`
    *   **Response**: GeoJSON-like path coordinates and metrics (Time, Pollution Exposure).
*   `POST /api/snap`: Snaps a clicked coordinate to the nearest valid road node.

---

## 👥 Contributors

*   **Team Auralis**
*   *Built for Smart India Hackathon (SIH)*
