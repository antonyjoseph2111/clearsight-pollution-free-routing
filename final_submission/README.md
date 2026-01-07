# Pollution-Free Routing System (Government Submission)

This submission package contains two versions of the application to demonstrate scalability and robust performance across different environments.

## 📂 Version 1: Pro Extended (Recommended)
**Path:** `Version_1_Pro_Extended/`

The full-featured version designed for real-world deployment covering the entire **National Capital Territory (NCT) of Delhi**.

*   **Coverage:** Entire Delhi NCR (Major Roads: Motorway to Tertiary).
*   **Technology:** Implements an intelligent "Sparse Graph" technique to filter minor roads, reducing memory usage by ~80% while maintaining city-wide connectivity.
*   **Resilience:** Includes an automatic **Fallback Mechanism**. If the server detects specialized low-memory constraints (e.g., <512MB RAM), it gracefully reverts to a smaller radius to prevent crashes.
*   **Use Case:** Full government pilots, extensive testing, and "Pro" tier deployments.

---

## 📂 Version 2: Showcase Lite
**Path:** `Version_2_Showcase_Lite/`

A lightweight, high-performance version optimized for **instant demonstrations** and extremely constrained environments.

*   **Coverage:** Fixed **2km Radius** around Connaught Place, New Delhi.
*   **Technology:** Pre-configured to download only a small, specific area. Zero startup overhead and minimal memory footprint.
*   **Performance:** Fastest boot time and guaranteed stability on any hardware (even <256MB RAM).
*   **Use Case:** Quick presentations, jury showcases, and ensuring functionality on very low-tier hosting plans without configuration.

---

## 🚀 Deployment Instructions

Both versions use the same standard Python/Flask stack.

### Local Run
1.  Navigate to the desired version folder:
    ```bash
    cd Version_1_Pro_Extended  # or Version_2_Showcase_Lite
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the application:
    ```bash
    python -m app.app
    ```

### Cloud Deployment (Render.com / AWS / Azure)
*   **Build Command:** `pip install -r requirements.txt`
*   **Start Command:** `gunicorn app:app --timeout 180`
*   **Python Version:** `3.10.0`

> **Note:** For live deployment, `Version_1_Pro_Extended` is recommended as it offers the best balance of coverage and stability. Use `Version_2_Showcase_Lite` if you encounter severe resource limitations or need an instant demo environment.

---

## 🏛️ System Architecture

1.  **Frontend (UI):**
    *   **HTML5/CSS3:** Mobile-responsive design with "bottom-sheet" controls for a native app feel.
    *   **Leaflet.js:** Interactive mapping and route visualization.
    *   **JavaScript (Vanilla):** Handles user geolocation, API calls, and dynamic map updates.

2.  **Backend (API):**
    *   **Flask (Python):** RESTful API server.
    *   **OSMnx & NetworkX:** Core routing engine. Handles graph construction, road data, and shortest path algorithms.
    *   **Scikit-Learn/TensorFlow:** (Optional) Integration points for pollution prediction models.

3.  **Data Layer:**
    *   **OpenStreetMap (OSM):** Road network data.
    *   **Open-Meteo API:** Real-time weather data for pollution forecasting.
    *   **GraphML Cache:** efficient storage of road networks to minimize startup time.

---

## 🛠️ Troubleshooting

**Issue: `gunicorn: command not found`**
*   Ensure `gunicorn` is in your `requirements.txt` (it is included by default).
*   Check your virtual environment activation.

**Issue: "Worker Timeout" on Render**
*   This happens if the graph download takes >30 seconds.
*   **Fix:** Ensure `--timeout 180` is in your Start Command.
*   **Fix:** Switch to `Version_2_Showcase_Lite` for instant startup.

**Issue: "MemoryError" or "Killed" process**
*   The server ran out of RAM (common on 512MB instances).
*   **Fix:** The application handles this automatically in Version 1 by fallback.
*   **Fix:** Use `Version_2_Showcase_Lite`.

---

## 📞 Contact & Support
For queries regarding the **Pollution-Free Routing System** submission:
*   **Developer:** [Your Name/Team Name]
*   **Project:** SIH 2024 Submission (ClearSight)

