from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
import folium
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import networkx as nx
from datetime import datetime, timedelta
from functools import lru_cache

app = Flask(__name__)

# Sample data for Indian railway stations
STATIONS = {
    "NDLS": {"name": "New Delhi", "lat": 28.6139, "lon": 77.2090, "importance": 10},
    "BCT": {"name": "Mumbai Central", "lat": 18.9712, "lon": 72.8190, "importance": 9},
    "MAS": {"name": "Chennai Central", "lat": 13.0827, "lon": 80.2707, "importance": 9},
    "HWH": {"name": "Howrah", "lat": 22.5958, "lon": 88.2636, "importance": 9},
    "SBC": {"name": "Bengaluru", "lat": 12.9784, "lon": 77.5996, "importance": 8},
    "JP": {"name": "Jaipur", "lat": 26.9124, "lon": 75.7873, "importance": 7},
    "ADI": {"name": "Ahmedabad", "lat": 23.0225, "lon": 72.5714, "importance": 7},
    "PUNE": {"name": "Pune", "lat": 18.5204, "lon": 73.8567, "importance": 7},
    "LKO": {"name": "Lucknow", "lat": 26.8467, "lon": 80.9462, "importance": 6},
    "CNB": {"name": "Kanpur", "lat": 26.4499, "lon": 80.3319, "importance": 6},
    "ALD": {"name": "Prayagraj", "lat": 25.4358, "lon": 81.8463, "importance": 6},
    "VSKP": {"name": "Visakhapatnam", "lat": 17.6868, "lon": 83.2185, "importance": 5},
    "BBS": {"name": "Bhubaneswar", "lat": 20.2961, "lon": 85.8245, "importance": 5},
    "PNBE": {"name": "Patna", "lat": 25.5941, "lon": 85.1376, "importance": 6},
    "GHY": {"name": "Guwahati", "lat": 26.1445, "lon": 91.7362, "importance": 5}
}

CONNECTIONS = [
    ("NDLS", "CNB", 435, 5.0, "Shatabdi Express"),
    ("CNB", "ALD", 200, 2.5, "Prayagraj Express"),
    ("ALD", "PNBE", 560, 7.0, "Ganga Express"),
    ("PNBE", "HWH", 535, 8.0, "Howrah Mail"),
    ("HWH", "GHY", 1020, 16.0, "Northeast Express"),
    ("NDLS", "JP", 310, 4.0, "Pink City Express"),
    ("JP", "ADI", 650, 10.0, "Ashram Express"),
    ("ADI", "BCT", 500, 7.0, "Gujarat Mail"),
    ("BCT", "PUNE", 190, 3.5, "Deccan Express"),
    ("PUNE", "SBC", 840, 14.0, "Udyan Express"),
    ("SBC", "MAS", 350, 5.0, "Shatabdi Express"),
    ("MAS", "VSKP", 800, 12.0, "Coromandel Express"),
    ("VSKP", "BBS", 445, 7.0, "East Coast Express"),
    ("BBS", "HWH", 440, 6.0, "Falaknuma Express"),
    ("CNB", "LKO", 75, 1.5, "Lucknow Intercity"),
    ("LKO", "NDLS", 500, 6.5, "Lucknow Mail"),
    ("ALD", "SBC", 1600, 26.0, "Karnataka Express"),
    ("NDLS", "ADI", 920, 14.0, "Rajdhani Express"),
    ("BCT", "MAS", 1280, 21.0, "Mumbai Mail"),
    # Additional connections to ensure all stations are connected
    ("MAS", "HWH", 1660, 28.0, "Howrah Mail"),
    ("SBC", "PUNE", 840, 14.0, "Udyan Express"),
    ("VSKP", "MAS", 800, 12.0, "Coromandel Express"),
    ("GHY", "PNBE", 580, 10.0, "Northeast Express"),
    ("PNBE", "NDLS", 990, 14.0, "Rajdhani Express"),
    ("BBS", "PNBE", 500, 8.0, "Purushottam Express"),
    ("LKO", "ALD", 200, 3.0, "Prayagraj Express"),
    ("JP", "NDLS", 310, 4.0, "Pink City Express"),
    ("ADI", "JP", 650, 10.0, "Ashram Express"),
    ("PUNE", "BCT", 190, 3.5, "Deccan Express")
]

def create_railway_graph():
    G = nx.Graph()
    for code, data in STATIONS.items():
        G.add_node(code, name=data["name"], lat=data["lat"], lon=data["lon"], importance=data.get("importance", 5))
    for src, dst, dist, time, train in CONNECTIONS:
        G.add_edge(src, dst, distance=dist, time=time, train=train)
    return G

railway_graph = create_railway_graph()

def train_delay_model():
    np.random.seed(42)
    n_samples = 2000
    distances = np.random.uniform(50, 2000, n_samples)
    hours = np.random.randint(0, 24, n_samples)
    days = np.random.randint(0, 7, n_samples)
    months = np.random.randint(1, 13, n_samples)
    src_importance = np.random.uniform(5, 10, n_samples)
    dst_importance = np.random.uniform(5, 10, n_samples)

    base_delays = distances * 0.05
    time_factor = np.ones(n_samples)
    morning_rush = (hours >= 7) & (hours <= 10)
    evening_rush = (hours >= 17) & (hours <= 20)
    night_time = (hours >= 22) | (hours <= 4)
    time_factor[morning_rush] = 1.5
    time_factor[evening_rush] = 1.8
    time_factor[night_time] = 0.7

    weekend = (days >= 5)
    day_factor = np.ones(n_samples)
    day_factor[weekend] = 1.3

    monsoon = (months >= 6) & (months <= 9)
    winter = (months == 12) | (months <= 2)
    seasonal_factor = np.ones(n_samples)
    seasonal_factor[monsoon] = 1.5
    seasonal_factor[winter] = 1.3

    station_factor = 2.0 - (src_importance + dst_importance) / 20
    delays = base_delays * time_factor * day_factor * seasonal_factor * station_factor
    delays += np.random.exponential(10, n_samples)
    delays = np.maximum(0, delays)

    X = pd.DataFrame({
        'distance': distances,
        'hour': hours,
        'day': days,
        'month': months,
        'src_importance': src_importance,
        'dst_importance': dst_importance
    })
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
    model.fit(X_scaled, delays)
    return model, scaler

delay_model, delay_scaler = train_delay_model()

def predict_delay(src, dst, departure_time):
    if not nx.has_path(railway_graph, src, dst):
        return 0
    path = nx.shortest_path(railway_graph, src, dst, weight='distance')
    total_distance = sum(railway_graph[path[i]][path[i+1]]['distance'] for i in range(len(path) - 1))

    hour = departure_time.hour
    day = departure_time.weekday()
    month = departure_time.month

    src_importance = railway_graph.nodes[src].get('importance', 5)
    dst_importance = railway_graph.nodes[dst].get('importance', 5)

    features = pd.DataFrame({
        'distance': [total_distance],
        'hour': [hour],
        'day': [day],
        'month': [month],
        'src_importance': [src_importance],
        'dst_importance': [dst_importance]
    })

    features_scaled = delay_scaler.transform(features)
    predicted_delay = delay_model.predict(features_scaled)[0]
    return predicted_delay

@lru_cache(maxsize=128)
def get_cached_route(src, dst, departure_time_str, priority):
    departure_time = datetime.strptime(departure_time_str, '%Y-%m-%dT%H:%M')
    return find_optimal_route(src, dst, departure_time, priority)

def find_optimal_route(src, dst, departure_time, priority='balanced'):
    if not nx.has_path(railway_graph, src, dst):
        return {'error': 'No path found'}

    if priority == 'distance':
        weight = 'distance'
    elif priority == 'time':
        weight = 'time'
    else:
        weight = 'balanced'

    try:
        path = nx.shortest_path(railway_graph, src, dst, weight=weight)
    except Exception:
        path = [src, dst]

    total_distance = 0
    total_time = 0
    route_segments = []
    trains_used = []

    for i in range(len(path) - 1):
        edge_data = railway_graph.get_edge_data(path[i], path[i+1]) or {}
        segment_distance = edge_data.get('distance', 1000)
        segment_time = edge_data.get('time', 20)
        train_name = edge_data.get('train', "Express")

        if train_name not in trains_used:
            trains_used.append(train_name)

        total_distance += segment_distance
        total_time += segment_time
        route_segments.append({
            'from': {**STATIONS[path[i]]},
            'to': {**STATIONS[path[i+1]]},
            'distance': segment_distance,
            'time': segment_time,
            'train': train_name
        })

    predicted_delay = predict_delay(src, dst, departure_time)
    arrival_time = departure_time + timedelta(hours=total_time) + timedelta(minutes=predicted_delay)
    base_fare = total_distance * 1.5
    ac_fare = base_fare * 2.5

    return {
        'path': path,
        'route_segments': route_segments,
        'total_distance': total_distance,
        'total_time': total_time,
        'predicted_delay': predicted_delay,
        'departure_time': departure_time.strftime('%Y-%m-%d %H:%M'),
        'arrival_time': arrival_time.strftime('%Y-%m-%d %H:%M'),
        'trains': trains_used,
        'fare': {
            'sleeper': round(base_fare),
            'ac_three_tier': round(ac_fare),
            'ac_two_tier': round(ac_fare * 1.4),
            'ac_first_class': round(ac_fare * 2)
        }
    }

@app.route('/')
def index():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <title>Indian Railway Route Planner</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"  />
    <style>
        body { margin: 0; font-family: Arial, sans-serif; background-color: #f4f4f4; }
        .container { display: flex; height: 100vh; }
        .sidebar { width: 300px; background: white; padding: 20px; box-shadow: 2px 0 5px rgba(0,0,0,0.1); overflow-y: auto; }
        .main-content { flex: 1; position: relative; }
        #map { width: 100%; height: 100%; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        select, input { width: 100%; padding: 10px; border-radius: 4px; border: 1px solid #ccc; }
        button { padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
        .tabs { display: flex; margin-bottom: 10px; }
        .tab { padding: 10px 15px; cursor: pointer; }
        .tab.active { background: #007bff; color: white; }
        .route-info { margin-top: 20px; }
        .segment { margin-bottom: 10px; border-left: 3px solid #007bff; padding-left: 10px; }
        .fare-details { margin-top: 20px; background: #fff; padding: 10px; border-radius: 4px; }
        .history-list { margin-top: 20px; }
        .history-item { padding: 10px; border-bottom: 1px solid #ddd; cursor: pointer; }
        .history-item:hover { background-color: #f0f0f0; }
    </style>
</head>
<body>
<div class="container">
    <div class="sidebar">
        <h2>Railway Route Planner</h2>
        <div class="tabs">
            <div class="tab active" onclick="switchTab('route')">Route Planner</div>
        </div>
        <div id="status-message"></div>
        <div id="route-planner" class="tab-content active">
            <form id="route-form">
                <label for="src">Origin Station:</label>
                <select id="src-station" required></select><br>
                <label for="dst">Destination Station:</label>
                <select id="dst-station" required></select><br>
                <label for="departure-time">Departure Time:</label>
                <input type="datetime-local" id="departure-time" required><br>
                <label for="priority">Route Priority:</label>
                <select id="priority">
                    <option value="balanced">Balanced</option>
                    <option value="distance">Shortest Distance</option>
                    <option value="time">Shortest Time</option>
                </select><br>
                <button type="submit">Find Route</button>
            </form>
            <div id="route-details" style="display:none;">
                <h3>Route Details</h3>
                <div id="summary"></div>
                <div id="segments"></div>
                <div id="fares"></div>
            </div>
            <div class="history-list" id="route-history">
                <h3>Route History</h3>
                <ul id="history-items"></ul>
            </div>
        </div>
    </div>
    <div class="main-content">
        <div id="map"></div>
    </div>
</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script> 
<script>
    let map;
    let currentRoute;
    let history = [];

    document.addEventListener('DOMContentLoaded', function () {
        initMap();
        loadStations();
        document.getElementById('route-form').addEventListener('submit', findRoute);
        document.getElementById('route-history').addEventListener('click', handleHistoryClick);
    });

    function initMap() {
        map = L.map('map').setView([22.5937, 78.9629], 5);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',  {
            attribution: '&copy; OpenStreetMap contributors'
        }).addTo(map);
    }

    function loadStations() {
        fetch('/get_stations').then(res => res.json()).then(data => {
            const srcSelect = document.getElementById('src-station');
            const dstSelect = document.getElementById('dst-station');

            srcSelect.innerHTML = '<option value="">Select Origin</option>';
            dstSelect.innerHTML = '<option value="">Select Destination</option>';

            for (const [code, info] of Object.entries(data)) {
                const option = document.createElement('option');
                option.value = code;
                option.textContent = `${info.name} (${code})`;
                srcSelect.appendChild(option.cloneNode(true));
                dstSelect.appendChild(option.cloneNode(true));
            }
        });
    }

    function switchTab(tabId) {
        document.querySelectorAll('.tab-content').forEach(el => el.style.display = 'none');
        document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
        document.getElementById(tabId).style.display = 'block';
        document.querySelector(`.tab[onclick="switchTab('${tabId}')"]`).classList.add('active');
    }

    function findRoute(e) {
        e.preventDefault();
        const src = document.getElementById('src-station').value;
        const dst = document.getElementById('dst-station').value;
        const time = document.getElementById('departure-time').value;
        const priority = document.getElementById('priority').value;

        if (!src || !dst || !time) {
            alert('Please fill all fields.');
            return;
        }

        fetch('/find_route', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ src, dst, departure_time: time, priority })
        }).then(res => res.json()).then(route => {
            if (route.error) {
                alert(route.error);
                return;
            }
            currentRoute = route;
            showRouteDetails(route);
            updateMapWithRoute(route);
            addToHistory(route);
        });
    }

    function showRouteDetails(route) {
        const summary = document.getElementById('summary');
        const segments = document.getElementById('segments');
        const fares = document.getElementById('fares');

        summary.innerHTML = `
            <p>Total Distance: ${route.total_distance.toFixed(1)} km</p>
            <p>Total Time: ${route.total_time.toFixed(1)} hours</p>
            <p>Predicted Delay: ${route.predicted_delay.toFixed(1)} minutes</p>
            <p>Departure: ${route.departure_time}</p>
            <p>Arrival: ${route.arrival_time}</p>
            <p>Trains: ${route.trains.join(', ')}</p>
        `;

        segments.innerHTML = '';
        route.route_segments.forEach((seg, i) => {
            segments.innerHTML += `<div class="segment">${i+1}. ${seg.from.name} → ${seg.to.name} (${seg.distance} km, ${seg.time} hrs by ${seg.train})</div>`;
        });

        fares.innerHTML = `
            <h4>Fares</h4>
            <p>Sleeper: ₹${route.fare.sleeper}</p>
            <p>AC Three Tier: ₹${route.fare.ac_three_tier}</p>
            <p>AC Two Tier: ₹${route.fare.ac_two_tier}</p>
            <p>AC First Class: ₹${route.fare.ac_first_class}</p>
        `;

        document.getElementById('route-details').style.display = 'block';
    }

    function updateMapWithRoute(route) {
        fetch('/get_route_map', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ route })
        }).then(res => res.text()).then(html => {
            document.getElementById('map').innerHTML = html;
        });
    }

    function addToHistory(route) {
        const item = document.createElement('li');
        item.className = 'history-item';
        item.textContent = `${route.route_segments[0].from.name} → ${route.route_segments[route.route_segments.length - 1].to.name}`;
        item.dataset.route = JSON.stringify(route);
        document.getElementById('history-items').appendChild(item);
    }

    function handleHistoryClick(e) {
        const target = e.target.closest('.history-item');
        if (!target) return;
        const route = JSON.parse(target.dataset.route);
        showRouteDetails(route);
        updateMapWithRoute(route);
    }

    function getStationName(code) {
        const stations = JSON.parse(localStorage.getItem('stations') || '{}');
        return stations[code]?.name || code;
    }

    window.onload = () => {
        localStorage.setItem('stations', JSON.stringify(STATIONS));
    };
</script>
</body>
</html>
"""

@app.route('/find_route', methods=['POST'])
def find_route():
    data = request.get_json()
    src = data.get('src')
    dst = data.get('dst')
    departure_str = data.get('departure_time')
    priority = data.get('priority', 'balanced')

    cached_route = get_cached_route(src, dst, departure_str, priority)
    return jsonify(cached_route)

@app.route('/get_map')
def get_map():
    m = create_railway_map()
    return m._repr_html_()

@app.route('/get_route_map', methods=['POST'])
def get_route_map():
    data = request.get_json()
    route = data.get('route')
    m = create_railway_map(route)
    return m._repr_html_()

@app.route('/get_stations')
def get_stations():
    return jsonify(STATIONS)

def create_railway_map(route=None):
    m = folium.Map(location=[22.5937, 78.9629], zoom_start=5, tiles='CartoDB positron')

    for code, data in STATIONS.items():
        popup_text = f"{data['name']} ({code})"
        icon_color = 'red' if data.get('importance', 5) >= 8 else 'blue' if data.get('importance', 5) >= 6 else 'green'
        folium.Marker(
            [data['lat'], data['lon']],
            popup=popup_text,
            icon=folium.Icon(icon='train', prefix='fa', color=icon_color)
        ).add_to(m)

    for src, dst, _, _, train in CONNECTIONS:
        src_lat, src_lon = STATIONS[src]['lat'], STATIONS[src]['lon']
        dst_lat, dst_lon = STATIONS[dst]['lat'], STATIONS[dst]['lon']
        popup_text = f"{STATIONS[src]['name']} to {STATIONS[dst]['name']}<br>Train: {train}"
        folium.PolyLine(
            [[src_lat, src_lon], [dst_lat, dst_lon]],
            popup=popup_text,
            color='gray',
            weight=2,
            opacity=0.7
        ).add_to(m)

    if route:
        route_coords = []
        for seg in route['route_segments']:
            src_lat, src_lon = seg['from']['lat'], seg['from']['lon']
            dst_lat, dst_lon = seg['to']['lat'], seg['to']['lon']
            route_coords.append([src_lat, src_lon])
            folium.PolyLine(
                [[src_lat, src_lon], [dst_lat, dst_lon]],
                color='red',
                weight=4,
                opacity=0.8
            ).add_to(m)
        route_coords.append([route['route_segments'][-1]['to']['lat'], route['route_segments'][-1]['to']['lon']])
        m.fit_bounds(route_coords)

    return m

if __name__ == '__main__':
    app.run(debug=True)
