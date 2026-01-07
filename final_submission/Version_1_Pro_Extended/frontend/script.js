// script.js
// Pollution-aware routing frontend: map, snapping, search, heatmap, TomTom refresh, routing.

const API_ROUTE = '/api/route';
const API_SNAP = '/api/snap';
const API_POLLUTION = '/api/pollution_points';
const API_TRAFFIC_REFRESH = '/api/traffic_refresh';
const API_TRAFFIC_STATUS = '/api/traffic_status';

const DEFAULT_CENTER = [28.6139, 77.2090];
const DEFAULT_ZOOM = 13;

let map;
let markerStart = null;
let markerEnd = null;
let routeLayers = [];
let heatLayerGroup = null;
let nextClickTarget = 'start'; // first click sets start, then end, then toggles

// UI references
const startInput = () => document.getElementById('start_input');
const endInput = () => document.getElementById('end_input');
const weightSlider = () => document.getElementById('weight_slider');
const weightValueEl = () => document.getElementById('weight_value');
const routeButton = () => document.getElementById('route_button');
const refreshButton = () => document.getElementById('refresh_traffic');
const messageBox = () => document.getElementById('message');
const resultsPanel = () => document.getElementById('results');
const metricsTable = () => document.getElementById('metrics_table');

// ---------------- UTILITY ----------------
function showMessage(txt, isError = true) {
    const mb = messageBox();
    if (!mb) return;
    mb.textContent = txt;
    mb.classList.remove('hidden');
    mb.classList.toggle('text-red-600', isError);
    mb.classList.toggle('text-green-600', !isError);
}

function hideMessage() {
    const mb = messageBox();
    if (!mb) return;
    mb.classList.add('hidden');
}

// convert slider 0-100 -> 0.00-1.00 and update display
function updateWeightDisplay(value) {
    const w = (parseInt(value, 10) / 100.0);
    const el = weightValueEl();
    if (el) el.textContent = w.toFixed(2);
    return w;
}

function enableRouteButtonIfReady() {
    const btn = routeButton();
    if (!btn) return;
    btn.disabled = !(markerStart && markerEnd);
}

// ---------------- MAP & MARKERS ----------------
function initializeMap() {
    map = L.map('map').setView(DEFAULT_CENTER, DEFAULT_ZOOM);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors',
        maxZoom: 19
    }).addTo(map);

    // heat layer group
    heatLayerGroup = L.layerGroup().addTo(map);

    // single click handler (alternating start/end)
    map.on('click', onMapClick);

    // wire UI controls
    if (weightSlider()) {
        weightSlider().addEventListener('input', (e) => updateWeightDisplay(e.target.value));
        // initialize display
        updateWeightDisplay(weightSlider().value);
    }

    if (routeButton()) {
        routeButton().addEventListener('click', (e) => { e.preventDefault(); findRoutes(); });
    }
    if (refreshButton()) {
        refreshButton().addEventListener('click', (e) => { e.preventDefault(); refreshLiveTraffic(); });
    }

    addSearchBox();
    loadPollutionHeatmap();

    // create helpful small legend/controls area buttons
    addCenterButtons();
}

// place or move marker and wire dragend snapping
function placeOrMoveMarker(which, latlng) {
    if (which === 'start') {
        if (!markerStart) {
            markerStart = L.marker(latlng, { draggable: true }).addTo(map).bindTooltip('Start', { permanent: true });
            markerStart.on('dragend', async () => {
                const p = markerStart.getLatLng();
                const snapped = await snapToServer(p.lat, p.lng);
                markerStart.setLatLng([snapped.lat, snapped.lon]);
                updateInput('start');
            });
        } else {
            markerStart.setLatLng(latlng);
        }
        updateInput('start');
    } else {
        if (!markerEnd) {
            markerEnd = L.marker(latlng, { draggable: true }).addTo(map).bindTooltip('End', { permanent: true });
            markerEnd.on('dragend', async () => {
                const p = markerEnd.getLatLng();
                const snapped = await snapToServer(p.lat, p.lng);
                markerEnd.setLatLng([snapped.lat, snapped.lon]);
                updateInput('end');
            });
        } else {
            markerEnd.setLatLng(latlng);
        }
        updateInput('end');
    }
    enableRouteButtonIfReady();
}

function updateInput(which) {
    if (which === 'start' && markerStart) {
        const p = markerStart.getLatLng();
        startInput().value = `${p.lat.toFixed(6)}, ${p.lng.toFixed(6)}`;
    }
    if (which === 'end' && markerEnd) {
        const p = markerEnd.getLatLng();
        endInput().value = `${p.lat.toFixed(6)}, ${p.lng.toFixed(6)}`;
    }
}

// map click -> snap -> place marker
async function onMapClick(e) {
    hideMessage();
    const lat = e.latlng.lat, lon = e.latlng.lng;
    console.log(`Map clicked at ${lat}, ${lon}`);
    let snapped;
    try {
        snapped = await snapToServer(lat, lon);
    } catch (err) {
        console.warn('Snap failed, using raw coords', err);
        snapped = { lat, lon };
    }
    placeOrMoveMarker(nextClickTarget, [snapped.lat, snapped.lon]);
    // alternate target
    nextClickTarget = (nextClickTarget === 'start') ? 'end' : 'start';
}

// snap to server helper
async function snapToServer(lat, lon) {
    try {
        const resp = await fetch(API_SNAP, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ lat, lon })
        });
        if (!resp.ok) {
            const txt = await resp.text().catch(()=>'');
            throw new Error(`Snap API ${resp.status} ${txt}`);
        }
        return await resp.json();
    } catch (e) {
        console.warn('snapToServer error', e);
        throw e;
    }
}

// ---------------- SEARCH / AUTOCOMPLETE (Nominatim) ----------------
function addSearchBox() {
    const controls = document.getElementById('controls');
    if (!controls) return;

    const wrapper = document.createElement('div');
    wrapper.className = 'pt-2';
    wrapper.innerHTML = `
      <input id="search_box" placeholder="Search place (e.g. India Gate)" 
             class="w-full p-2 border border-gray-300 rounded-lg" />
      <div id="search_results" class="bg-white shadow rounded mt-1 max-h-48 overflow-auto"></div>
    `;
    controls.insertBefore(wrapper, controls.firstChild ? controls.firstChild.nextSibling : controls.firstChild);

    const box = document.getElementById('search_box');
    const results = document.getElementById('search_results');
    if (!box || !results) return;

    let timer = null;
    box.addEventListener('input', (e) => {
        const q = e.target.value.trim();
        results.innerHTML = '';
        if (timer) clearTimeout(timer);
        if (q.length < 3) return;
        timer = setTimeout(() => nominatimSearch(q, results), 300);
    });
}

async function nominatimSearch(q, resultsEl) {
const url = `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(q)}&addressdetails=1&limit=6&countrycodes=in`;
    try {
        const r = await fetch(url, { headers: { 'Accept-Language': 'en' }});
        if (!r.ok) return;
        const data = await r.json();
        resultsEl.innerHTML = data.map(item => {
            return `<div class="p-2 hover:bg-gray-100 cursor-pointer" data-lat="${item.lat}" data-lon="${item.lon}">${item.display_name}</div>`;
        }).join('');
        resultsEl.querySelectorAll('div').forEach(div => {
            div.addEventListener('click', async () => {
                const lat = parseFloat(div.dataset.lat), lon = parseFloat(div.dataset.lon);
                map.setView([lat, lon], 16);
                try {
                    const snapped = await snapToServer(lat, lon);
                    placeOrMoveMarker(nextClickTarget, [snapped.lat, snapped.lon]);
                } catch {
                    placeOrMoveMarker(nextClickTarget, [lat, lon]);
                }
                nextClickTarget = (nextClickTarget === 'start') ? 'end' : 'start';
                resultsEl.innerHTML = '';
                document.getElementById('search_box').value = '';
            });
        });
    } catch (e) {
        console.warn('Nominatim search failed', e);
    }
}

// ---------------- HEATMAP / POLLUTION POINTS ----------------
async function loadPollutionHeatmap() {
    heatLayerGroup.clearLayers();
    try {
        const resp = await fetch(API_POLLUTION);
        if (!resp.ok) return;
        const pts = await resp.json();
        pts.forEach(pt => {
            const c = L.circleMarker([pt.lat, pt.lon], {
                radius: Math.max(5, Math.min(25, pt.ps / 40)),
                fillColor: getColorForPS(pt.ps),
                color: null,
                fillOpacity: 0.5
            }).addTo(heatLayerGroup);
            c.bindTooltip(`PS: ${Math.round(pt.ps)}`, { permanent: false });
        });
    } catch (e) {
        console.warn('Failed to load pollution points', e);
    }
}

function getColorForPS(ps) {
    const v = Math.min(Math.max(ps, 0), 500) / 500;
    const r = Math.round(255 * v);
    const g = Math.round(255 * (1 - v));
    return `rgb(${r},${g},50)`;
}

// ---------------- ROUTING / DRAW ----------------
function clearRoutes() {
    routeLayers.forEach(l => map.removeLayer(l));
    routeLayers = [];
}

function drawRoute(coords, color, label) {
    if (!coords || coords.length === 0) return;
    const latlngs = coords.map(c => [c[0], c[1]]);
    const poly = L.polyline(latlngs, { color, weight: 5, opacity: 0.9, lineCap: 'round' }).addTo(map);
    poly.bindTooltip(label, { permanent: true, direction: 'center' });
    routeLayers.push(poly);
    if (routeLayers.length === 1) map.fitBounds(poly.getBounds(), { padding: [40, 40] });
}

function updateMetricsTable(metrics) {
    if (!metrics) return;
    const main = metrics.Main_Route || {};
    const fast = metrics.Fastest_Route || {};
    const w = weightValueEl() ? weightValueEl().textContent : '0.50';
    metricsTable().innerHTML = `
      <thead><tr>
        <th>Route Type</th><th>Time (min)</th><th>Exposure (PS-min)</th><th>Peak PS</th>
      </tr></thead>
      <tbody>
        <tr style="color:#E60000; font-weight:bold;">
          <td>Fastest (w=0)</td><td>${fast.Time_min ?? '-'}</td><td>${fast.Exposure_PS_min ?? '-'}</td><td>${fast.Peak_PS ?? '-'}</td>
        </tr>
        <tr style="color:#0070C0; font-weight:bold;">
          <td>Your Choice (w=${w})</td><td>${main.Time_min ?? '-'}</td><td>${main.Exposure_PS_min ?? '-'}</td><td>${main.Peak_PS ?? '-'}</td>
        </tr>
      </tbody>
    `;
    // show results panel
    const res = resultsPanel();
    if (res) res.classList.remove('hidden');
}

// ---------------- FIND ROUTES (call backend) ----------------
async function findRoutes() {
    clearRoutes();
    hideMessage();
    if (!markerStart || !markerEnd) { showMessage('Please select both start and end points', true); return; }

    const start = startInput().value.trim();
    const end = endInput().value.trim();
    const weight = parseFloat(weightValueEl().textContent);

    routeButton().disabled = true;
    document.getElementById('controls').classList.add('loading');

    try {
        const resp = await fetch(API_ROUTE, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ start, end, weight })
        });
        const data = await resp.json();
        if (!resp.ok) {
            const msg = data.error || data.detail || 'Server error';
            showMessage(`API Error: ${msg}`, true);
            return;
        }
        // draw routes
        drawRoute(data.fastest_route_coords, '#E60000', 'Fastest Route');
        drawRoute(data.main_route_coords, '#0070C0', 'Pollution-Optimized');
        updateMetricsTable(data.metrics);
        showMessage('Routes calculated', false);
        // reload heatmap in case pollution changed
        loadPollutionHeatmap();
    } catch (e) {
        console.error('findRoutes error', e);
        showMessage('Network or server error while calculating routes', true);
    } finally {
        routeButton().disabled = false;
        document.getElementById('controls').classList.remove('loading');
    }
}

// ---------------- TRAFFIC REFRESH ----------------
async function refreshLiveTraffic() {
    hideMessage();
    const btn = refreshButton();
    if (btn) { btn.disabled = true; btn.textContent = 'Refreshing traffic...'; }
    try {
        const resp = await fetch(API_TRAFFIC_REFRESH, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ max_points: 120, spacing_deg: 0.005 })
        });
        const data = await resp.json();
        if (!resp.ok || !data.ok) {
            showMessage('Traffic refresh failed: ' + (data.error || 'server error'), true);
        } else {
            showMessage('Traffic updated — updated edges: ' + (data.result.updated_edges || 0), false);
            // re-run current route if points set
            if (markerStart && markerEnd) findRoutes();
        }
    } catch (e) {
        console.error('refreshLiveTraffic error', e);
        showMessage('Network error refreshing traffic', true);
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = 'Refresh Live Traffic'; }
    }
}

// ---------------- HELPERS: center buttons ----------------
function addCenterButtons() {
    const controls = document.getElementById('controls');
    if (!controls) return;
    const btnContainer = document.createElement('div');
    btnContainer.style.marginTop = '8px';
    btnContainer.innerHTML = `
      <button id="set_start_center" class="w-full py-1 mb-1 bg-blue-500 text-white rounded text-sm">Set Start at Map Center</button>
      <button id="set_end_center" class="w-full py-1 bg-indigo-600 text-white rounded text-sm">Set End at Map Center</button>
    `;
    controls.appendChild(btnContainer);

    const startBtn = document.getElementById('set_start_center');
    const endBtn = document.getElementById('set_end_center');
    if (startBtn) startBtn.addEventListener('click', () => setAtCenter('start'));
    if (endBtn) endBtn.addEventListener('click', () => setAtCenter('end'));
}

async function setAtCenter(which) {
    try {
        const c = map.getCenter();
        const snapped = await snapToServer(c.lat, c.lng).catch(()=>({lat:c.lat, lon:c.lng}));
        placeOrMoveMarker(which, [snapped.lat, snapped.lon]);
        updateInput(which);
        enableRouteButtonIfReady();
        showMessage(`${which === 'start' ? 'Start' : 'End'} set at map center`, false);
    } catch (e) {
        console.error('setAtCenter error', e);
        showMessage('Failed to set point at center', true);
    }
}

// ---------------- BOOT ----------------
window.addEventListener('load', () => {
    initializeMap();
    // disable route button until both markers are set
    if (routeButton()) routeButton().disabled = true;
});
