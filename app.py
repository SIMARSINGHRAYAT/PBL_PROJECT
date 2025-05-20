from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import folium
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import networkx as nx
from datetime import datetime, timedelta
from functools import lru_cache

app = Flask(__name__)

# Sample data for Indian railway stations (simplified for demonstration)
# In a real application, this would come from a database
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

# Sample railway connections (edges in the graph)
CONNECTIONS = [
("NDLS", "CNB", 435, 5.0, "Shatabdi Express"),  # (station1, station2, distance in km, normal travel time in hours, train name)
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

# Create a graph for the railway network
def create_railway_graph():
    G = nx.Graph()
    
    # Add nodes (stations)
    for code, data in STATIONS.items():
        G.add_node(code, name=data["name"], lat=data["lat"], lon=data["lon"], importance=data.get("importance", 5))
    
    # Add edges (connections)
    for src, dst, dist, time, train in CONNECTIONS:
        G.add_edge(src, dst, distance=dist, time=time, train=train)
    
    # Ensure the graph is connected
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        print(f"Warning: Graph is not connected. Found {len(components)} components.")
        
        # If there are disconnected components, add connections between them
        if len(components) > 1:
            for i in range(len(components) - 1):
                comp1 = list(components[i])[0]  # Take first station from component
                comp2 = list(components[i + 1])[0]  # Take first station from next component
                
                # Calculate straight-line distance
                lat1, lon1 = STATIONS[comp1]["lat"], STATIONS[comp1]["lon"]
                lat2, lon2 = STATIONS[comp2]["lat"], STATIONS[comp2]["lon"]
                
                # Simple distance calculation (not accurate for long distances but sufficient for our purpose)
                dist = ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5 * 111  # Approx km per degree
                time = dist / 60  # Assume 60 km/h average speed
                
                # Add connection
                G.add_edge(comp1, comp2, distance=dist, time=time, train="Connection Express")
                CONNECTIONS.append((comp1, comp2, dist, time, "Connection Express"))
                print(f"Added connection between {comp1} and {comp2} to connect components")
    
    return G

# Initialize the graph
railway_graph = create_railway_graph()

# Train a more accurate ML model for delay prediction
def train_delay_model():
    # In a real application, this would use historical data
    # For demonstration, we'll create synthetic data with more realistic patterns
    np.random.seed(42)
    
    # Features: distance, time of day, day of week, month, importance of stations
    n_samples = 2000  # More samples for better training
    distances = np.random.uniform(50, 2000, n_samples)
    hours = np.random.randint(0, 24, n_samples)
    days = np.random.randint(0, 7, n_samples)  # 0=Monday, 6=Sunday
    months = np.random.randint(1, 13, n_samples)
    src_importance = np.random.uniform(5, 10, n_samples)  # Station importance (major stations have better on-time performance)
    dst_importance = np.random.uniform(5, 10, n_samples)
    
    # Target: delay in minutes (more delay for longer distances and certain times)
    # Base delay is proportional to distance
    base_delays = distances * 0.05  # 5 minutes per 100km on average
    
    # Time of day factors (rush hours have more delays)
    morning_rush = (hours >= 7) & (hours <= 10)
    evening_rush = (hours >= 17) & (hours <= 20)
    night_time = (hours >= 22) | (hours <= 4)
    
    time_factor = np.ones(n_samples)
    time_factor[morning_rush] = 1.5
    time_factor[evening_rush] = 1.8
    time_factor[night_time] = 0.7  # Less traffic at night, fewer delays
    
    # Day of week factors (weekends have different patterns)
    weekend = (days >= 5)  # Friday, Saturday
    day_factor = np.ones(n_samples)
    day_factor[weekend] = 1.3  # More passenger traffic on weekends
    
    # Seasonal factors (monsoon, winter fog, summer heat)
    monsoon = (months >= 6) & (months <= 9)  # June to September
    winter = (months == 12) | (months <= 2)  # December to February
    
    seasonal_factor = np.ones(n_samples)
    seasonal_factor[monsoon] = 1.5  # Monsoon causes delays
    seasonal_factor[winter] = 1.3  # Winter fog causes delays
    
    # Station importance factor (major stations have better on-time performance)
    station_factor = 2.0 - (src_importance + dst_importance) / 20  # Range: 0.5 to 1.5
    
    # Combine all factors
    delays = base_delays * time_factor * day_factor * seasonal_factor * station_factor
    
    # Add some randomness to simulate real-world variability
    delays += np.random.exponential(10, n_samples)  # Exponential distribution for more realistic delay patterns
    delays = np.maximum(0, delays)  # No negative delays
    
    # Create a DataFrame with all features
    X = pd.DataFrame({
        'distance': distances,
        'hour': hours,
        'day': days,
        'month': months,
        'src_importance': src_importance,
        'dst_importance': dst_importance
    })
    
    # Train a Random Forest model with more trees for better accuracy
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
    model.fit(X_scaled, delays)
    
    return model, scaler

# Initialize the ML model
delay_model, delay_scaler = train_delay_model()

# Predict delay for a given route with improved accuracy
def predict_delay(src, dst, departure_time):
    if not nx.has_path(railway_graph, src, dst):
        return 0  # Default to 0 delay if no path (should not happen with connected graph)
    
    path = nx.shortest_path(railway_graph, src, dst, weight='distance')
    total_distance = 0
    
    for i in range(len(path) - 1):
        total_distance += railway_graph[path[i]][path[i+1]]['distance']
    
    # Extract features for prediction
    hour = departure_time.hour
    day = departure_time.weekday()
    month = departure_time.month
    
    # Get station importance
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

# Cache for route calculations (lru_cache will store up to 128 recent route calculations)
@lru_cache(maxsize=128)
def get_cached_route(src, dst, departure_time_str, priority):
    # Convert string time to datetime for processing
    departure_time = datetime.strptime(departure_time_str, '%Y-%m-%dT%H:%M')
    return find_optimal_route(src, dst, departure_time, priority)

# Find optimal route between two stations with improved algorithm
def find_optimal_route(src, dst, departure_time, priority='balanced'):
    # Always ensure we can find a path
    if not nx.has_path(railway_graph, src, dst):
        # This should not happen with our connected graph, but just in case
        print(f"No direct path found between {src} and {dst}. Finding alternative route.")
        
        # Find the closest connected station to source and destination
        closest_to_src = src
        closest_to_dst = dst
        
        # If we still can't find a path, return a default route
        return {
            'path': [src, dst],
            'route_segments': [{
                'from': {
                    'code': src,
                    'name': STATIONS[src]['name'],
                    'lat': STATIONS[src]['lat'],
                    'lon': STATIONS[src]['lon']
                },
                'to': {
                    'code': dst,
                    'name': STATIONS[dst]['name'],
                    'lat': STATIONS[dst]['lat'],
                    'lon': STATIONS[dst]['lon']
                },
                'distance': 1000,  # Default distance
                'time': 20,  # Default time
                'train': "Direct Express"
            }],
            'total_distance': 1000,
            'total_time': 20,
            'predicted_delay': 30,
            'departure_time': departure_time.strftime('%Y-%m-%d %H:%M'),
            'arrival_time': (departure_time + timedelta(hours=20, minutes=30)).strftime('%Y-%m-%d %H:%M'),
            'trains': ["Direct Express"],
            'fare': {
                'sleeper': 1500,
                'ac_three_tier': 3750,
                'ac_two_tier': 5250,
                'ac_first_class': 7500
            }
        }
    
    # Different weights based on priority
    if priority == 'distance':
        weight = 'distance'
    elif priority == 'time':
        weight = 'time'
    elif priority == 'delay':
        # Create a copy of the graph with predicted delay as weight
        G_delay = railway_graph.copy()
        for u, v, data in G_delay.edges(data=True):
            # Estimate delay for this segment based on distance and time of day
            segment_distance = data['distance']
            segment_time = data['time']
            
            # Simple heuristic: longer segments and peak hours have more delay
            hour_factor = 1.5 if (departure_time.hour >= 8 and departure_time.hour <= 10) or \
                              (departure_time.hour >= 17 and departure_time.hour <= 19) else 1.0
            
            estimated_delay = segment_distance * 0.03 * hour_factor  # 3 minutes per 100km
            
            # Weight is time + estimated delay
            G_delay[u][v]['delay_weight'] = segment_time + (estimated_delay / 60)  # Convert delay to hours
        
        weight = 'delay_weight'
        temp_graph = G_delay
    else:  # balanced
        # Create a copy of the graph with balanced weights
        G_balanced = railway_graph.copy()
        for u, v, data in G_balanced.edges(data=True):
            # Normalize distance and time to be on similar scales
            # and combine them for a balanced approach
            G_balanced[u][v]['balanced'] = (data['distance']/100) + data['time']
        weight = 'balanced'
        temp_graph = G_balanced
    
    # Find the shortest path
    try:
        if priority == 'balanced' or priority == 'delay':
            path = nx.shortest_path(temp_graph, src, dst, weight=weight)
        else:
            path = nx.shortest_path(railway_graph, src, dst, weight=weight)
    except Exception as e:
        print(f"Error finding path: {e}")
        # Fallback to any path if optimal path fails
        try:
            path = nx.shortest_path(railway_graph, src, dst)
        except:
            # If all else fails, create a direct connection
            print(f"Creating direct connection between {src} and {dst}")
            
            # Calculate straight-line distance
            lat1, lon1 = STATIONS[src]["lat"], STATIONS[src]["lon"]
            lat2, lon2 = STATIONS[dst]["lat"], STATIONS[dst]["lon"]
            
            # Simple distance calculation
            dist = ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5 * 111  # Approx km per degree
            time = dist / 60  # Assume 60 km/h average speed
            
            # Add temporary connection
            railway_graph.add_edge(src, dst, distance=dist, time=time, train="Direct Express")
            
            # Now we should be able to find a path
            path = [src, dst]
    
    # Calculate route details
    total_distance = 0
    total_time = 0
    route_segments = []
    trains_used = []
    
    for i in range(len(path) - 1):
        src_code = path[i]
        dst_code = path[i+1]
        
        # Get edge data, with fallbacks in case the edge doesn't exist
        edge_data = railway_graph.get_edge_data(src_code, dst_code) or {}
        segment_distance = edge_data.get('distance', 1000)  # Default to 1000 km if missing
        segment_time = edge_data.get('time', 20)  # Default to 20 hours if missing
        train_name = edge_data.get('train', "Express")  # Default train name
        
        if train_name not in trains_used:
            trains_used.append(train_name)
        
        total_distance += segment_distance
        total_time += segment_time
        
        route_segments.append({
            'from': {
                'code': src_code,
                'name': STATIONS[src_code]['name'],
                'lat': STATIONS[src_code]['lat'],
                'lon': STATIONS[src_code]['lon']
            },
            'to': {
                'code': dst_code,
                'name': STATIONS[dst_code]['name'],
                'lat': STATIONS[dst_code]['lat'],
                'lon': STATIONS[dst_code]['lon']
            },
            'distance': segment_distance,
            'time': segment_time,
            'train': train_name
        })
    
    # Predict delay
    predicted_delay = predict_delay(src, dst, departure_time)
    
    # Calculate arrival time
    arrival_time = departure_time + timedelta(hours=total_time) + timedelta(minutes=predicted_delay)
    
    # Calculate fare (simplified model)
    base_fare = total_distance * 1.5  # Rs. 1.5 per km
    ac_fare = base_fare * 2.5  # AC class costs 2.5 times more
    
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

# Create a map with the railway network
def create_railway_map(route=None):
    # Create a map centered on India
    m = folium.Map(location=[22.5937, 78.9629], zoom_start=5, tiles='CartoDB positron')
    
    # Add station markers
    for code, data in STATIONS.items():
        popup_text = f"{data['name']} ({code})"
        
        # Use different colors based on station importance
        if data.get('importance', 5) >= 8:
            icon_color = 'red'
        elif data.get('importance', 5) >= 6:
            icon_color = 'blue'
        else:
            icon_color = 'green'
            
        folium.Marker(
            location=[data['lat'], data['lon']],
            popup=popup_text,
            icon=folium.Icon(icon='train', prefix='fa', color=icon_color)
        ).add_to(m)
    
    # Add railway connections
    for src, dst, _, _, train in CONNECTIONS:
        src_lat, src_lon = STATIONS[src]['lat'], STATIONS[src]['lon']
        dst_lat, dst_lon = STATIONS[dst]['lat'], STATIONS[dst]['lon']
        
        # Popup with connection details
        popup_text = f"{STATIONS[src]['name']} to {STATIONS[dst]['name']}<br>Train: {train}"
        
        folium.PolyLine(
            locations=[[src_lat, src_lon], [dst_lat, dst_lon]],
            popup=popup_text,
            color='gray',
            weight=2,
            opacity=0.7
        ).add_to(m)
    
    # Highlight the selected route if provided
    if route:
        route_coords = []
        for segment in route['route_segments']:
            src_lat, src_lon = segment['from']['lat'], segment['from']['lon']
            dst_lat, dst_lon = segment['to']['lat'], segment['to']['lon']
            route_coords.append([src_lat, src_lon])
            
            # Add a popup for each segment with train information
            popup_text = f"""
            <b>{segment['from']['name']} to {segment['to']['name']}</b><br>
            Train: {segment.get('train', 'Unknown')}<br>
            Distance: {segment['distance']} km<br>
            Time: {segment['time']} hours
            """
            
            folium.PolyLine(
                locations=[[src_lat, src_lon], [dst_lat, dst_lon]],
                popup=popup_text,
                color='red',
                weight=4,
                opacity=0.8
            ).add_to(m)
        
        # Add the last destination point
        route_coords.append([route['route_segments'][-1]['to']['lat'], route['route_segments'][-1]['to']['lon']])
        
        # Fit the map to the route
        m.fit_bounds(route_coords)
    
    return m

# Routes
@app.route('/')
def index():
    # Return the HTML directly instead of using render_template
    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Indian Railway Network Management System</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css" />
<style>
    :root {
        --primary-color: #0056b3;
        --secondary-color: #003d82;
        --accent-color: #FF5722;
        --light-bg: #f5f5f5;
        --card-bg: #fff;
        --text-color: #333;
        --border-color: #ddd;
        --success-color: #28a745;
        --warning-color: #ffc107;
        --danger-color: #dc3545;
    }
    
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-color: var(--light-bg);
        color: var(--text-color);
    }
    
    .container {
        display: flex;
        height: 100vh;
    }
    
    .sidebar {
        width: 350px;
        background-color: var(--card-bg);
        padding: 20px;
        box-shadow: 2px 0 5px rgba(0,0,0,0.1);
        overflow-y: auto;
        display: flex;
        flex-direction: column;
    }
    
    .main-content {
        flex: 1;
        position: relative;
    }
    
    .map-container {
        width: 100%;
        height: 100%;
        position: relative;
    }
    
    #map {
        width: 100%;
        height: 100%;
    }
    
    .header {
        display: flex;
        align-items: center;
        margin-bottom: 20px;
        padding-bottom: 10px;
        border-bottom: 2px solid var(--primary-color);
    }
    
    .header i {
        font-size: 24px;
        color: var(--primary-color);
        margin-right: 10px;
    }
    
    h1, h2, h3, h4 {
        color: var(--primary-color);
    }
    
    h1 {
        font-size: 1.5rem;
        margin: 0;
    }
    
    .form-group {
        margin-bottom: 15px;
    }
    
    label {
        display: block;
        margin-bottom: 5px;
        font-weight: bold;
    }
    
    select, input {
        width: 100%;
        padding: 10px;
        border: 1px solid var(--border-color);
        border-radius: 4px;
        font-size: 14px;
    }
    
    select:focus, input:focus {
        outline: none;
        border-color: var(--primary-color);
    }
    
    button {
        background-color: var(--primary-color);
        color: white;
        border: none;
        padding: 10px 15px;
        border-radius: 4px;
        cursor: pointer;
        font-weight: bold;
        transition: background-color 0.3s;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    button i {
        margin-right: 8px;
    }
    
    button:hover {
        background-color: var(--secondary-color);
    }
    
    .route-info {
        margin-top: 20px;
        border-top: 1px solid var(--border-color);
        padding-top: 15px;
    }
    
    .route-summary {
        background-color: var(--card-bg);
        padding: 15px;
        border-radius: 4px;
        margin-bottom: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    .route-segment {
        margin-bottom: 10px;
        padding: 12px;
        background-color: var(--card-bg);
        border-radius: 4px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        position: relative;
    }
    
    .route-segment:not(:last-child)::after {
        content: '';
        position: absolute;
        bottom: -10px;
        left: 20px;
        width: 2px;
        height: 10px;
        background-color: var(--accent-color);
    }
    
    .tabs {
        display: flex;
        margin-bottom: 15px;
        border-bottom: 1px solid var(--border-color);
    }
    
    .tab {
        padding: 10px 15px;
        cursor: pointer;
        border-bottom: 2px solid transparent;
        display: flex;
        align-items: center;
    }
    
    .tab i {
        margin-right: 5px;
    }
    
    .tab.active {
        border-bottom: 2px solid var(--primary-color);
        font-weight: bold;
    }
    
    .tab-content {
        display: none;
    }
    
    .tab-content.active {
        display: block;
    }
    
    .delay-warning {
        color: var(--danger-color);
        font-weight: bold;
    }
    
    .station-list {
        max-height: 200px;
        overflow-y: auto;
        margin-bottom: 15px;
        border: 1px solid var(--border-color);
        border-radius: 4px;
    }
    
    .station-item {
        padding: 10px;
        border-bottom: 1px solid var(--border-color);
        cursor: pointer;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .station-item:hover {
        background-color: rgba(0,0,0,0.03);
    }
    
    .station-item:last-child {
        border-bottom: none;
    }
    
    .loading {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-color: rgba(255,255,255,0.8);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 1000;
        display: none;
    }
    
    .spinner {
        width: 40px;
        height: 40px;
        border: 4px solid rgba(0,0,0,0.1);
        border-radius: 50%;
        border-top-color: var(--primary-color);
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    .fare-details {
        margin-top: 15px;
        background-color: var(--card-bg);
        padding: 15px;
        border-radius: 4px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    .fare-details h4 {
        margin-bottom: 10px;
        display: flex;
        align-items: center;
    }
    
    .fare-details h4 i {
        margin-right: 8px;
        color: var(--accent-color);
    }
    
    .fare-option {
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 1px solid var(--border-color);
    }
    
    .fare-option:last-child {
        border-bottom: none;
    }
    
    .fare-class {
        font-weight: 500;
    }
    
    .fare-amount {
        font-weight: 600;
        color: var(--primary-color);
    }
    
    .search-container {
        position: relative;
        margin-bottom: 15px;
    }
    
    .search-container input {
        padding-left: 35px;
    }
    
    .search-container i {
        position: absolute;
        left: 10px;
        top: 50%;
        transform: translateY(-50%);
        color: #999;
    }
    
    /* Responsive styles */
    @media (max-width: 768px) {
        .container {
            flex-direction: column;
        }
        
        .sidebar {
            width: 100%;
            height: 50%;
            overflow-y: auto;
        }
        
        .main-content {
            height: 50%;
        }
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--light-bg);
    }
    
    ::-webkit-scrollbar-thumb {
        background: #999;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-color);
    }
    
    .status-message {
        padding: 10px;
        margin-bottom: 15px;
        border-radius: 4px;
        display: none;
    }
    
    .status-success {
        background-color: rgba(40, 167, 69, 0.1);
        border: 1px solid var(--success-color);
        color: var(--success-color);
    }
    
    .status-error {
        background-color: rgba(220, 53, 69, 0.1);
        border: 1px solid var(--danger-color);
        color: var(--danger-color);
    }
    
    .status-warning {
        background-color: rgba(255, 193, 7, 0.1);
        border: 1px solid var(--warning-color);
        color: var(--warning-color);
    }
</style>
</head>
<body>
<div class="container">
    <div class="sidebar">
        <div class="header">
            <i class="fas fa-train"></i>
            <h1>Indian Railway Network</h1>
        </div>
        
        <div class="tabs">
            <div class="tab active" data-tab="route-planner">
                <i class="fas fa-route"></i> Route Planner
            </div>
            <div class="tab" data-tab="network-management">
                <i class="fas fa-cogs"></i> Network Management
            </div>
        </div>
        
        <div id="status-message" class="status-message"></div>
        
        <div class="tab-content active" id="route-planner">
            <form id="route-form">
                <div class="search-container">
                    <i class="fas fa-search"></i>
                    <input type="text" id="station-search" placeholder="Search for stations...">
                </div>
                
                <div class="form-group">
                    <label for="src-station">Origin Station:</label>
                    <select id="src-station" required>
                        <option value="">Select Origin</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="dst-station">Destination Station:</label>
                    <select id="dst-station" required>
                        <option value="">Select Destination</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="departure-time">Departure Time:</label>
                    <input type="datetime-local" id="departure-time" required>
                </div>
                
                <div class="form-group">
                    <label for="priority">Route Priority:</label>
                    <select id="priority">
                        <option value="balanced">Balanced</option>
                        <option value="distance">Shortest Distance</option>
                        <option value="time">Shortest Time</option>
                        <option value="delay">Minimize Delays</option>
                    </select>
                </div>
                
                <button type="submit"><i class="fas fa-search"></i> Find Route</button>
            </form>
            
            <div id="route-details" class="route-info" style="display: none;">
                <h3><i class="fas fa-map-marked-alt"></i> Route Details</h3>
                <div id="route-summary" class="route-summary"></div>
                <div id="route-segments"></div>
                <div id="fare-details" class="fare-details"></div>
            </div>
        </div>
        
        <div class="tab-content" id="network-management">
            <h3><i class="fas fa-subway"></i> Manage Stations</h3>
            
            <div class="search-container">
                <i class="fas fa-search"></i>
                <input type="text" id="station-list-search" placeholder="Search stations...">
            </div>
            
            <div class="station-list" id="station-list"></div>
            
            <form id="add-station-form">
                <h4><i class="fas fa-plus-circle"></i> Add New Station</h4>
                <div class="form-group">
                    <label for="station-code">Station Code:</label>
                    <input type="text" id="station-code" required maxlength="5">
                </div>
                <div class="form-group">
                    <label for="station-name">Station Name:</label>
                    <input type="text" id="station-name" required>
                </div>
                <div class="form-group">
                    <label for="station-lat">Latitude:</label>
                    <input type="number" id="station-lat" step="0.0001" required>
                </div>
                <div class="form-group">
                    <label for="station-lon">Longitude:</label>
                    <input type="number" id="station-lon" step="0.0001" required>
                </div>
                <div class="form-group">
                    <label for="station-importance">Importance (1-10):</label>
                    <input type="number" id="station-importance" min="1" max="10" value="5" required>
                </div>
                <button type="submit"><i class="fas fa-plus"></i> Add Station</button>
            </form>
            
            <form id="add-connection-form" style="margin-top: 20px;">
                <h4><i class="fas fa-link"></i> Add Connection</h4>
                <div class="form-group">
                    <label for="conn-src">From Station:</label>
                    <select id="conn-src" required>
                        <option value="">Select Station</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="conn-dst">To Station:</label>
                    <select id="conn-dst" required>
                        <option value="">Select Station</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="conn-distance">Distance (km):</label>
                    <input type="number" id="conn-distance" min="1" required>
                </div>
                <div class="form-group">
                    <label for="conn-time">Travel Time (hours):</label>
                    <input type="number" id="conn-time" step="0.1" min="0.1" required>
                </div>
                <div class="form-group">
                    <label for="conn-train">Train Name:</label>
                    <input type="text" id="conn-train" required>
                </div>
                <button type="submit"><i class="fas fa-plus"></i> Add Connection</button>
            </form>
        </div>
    </div>
    
    <div class="main-content">
        <div class="map-container">
            <div id="map"></div>
            <div class="loading" id="map-loading">
                <div class="spinner"></div>
            </div>
        </div>
    </div>
</div>

<script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
<script>
    // Initialize variables
    let map;
    let stations = {};
    let currentRoute = null;
    let stationMarkers = {};
    let connectionLines = [];
    let routeLine = null;
    
    // Initialize the application
    document.addEventListener('DOMContentLoaded', function() {
        // Set default departure time to now + 1 hour
        const now = new Date();
        now.setHours(now.getHours() + 1);
        document.getElementById('departure-time').value = now.toISOString().slice(0, 16);
        
        // Initialize map
        initMap();
        
        // Load stations
        loadStations();
        
        // Set up event listeners
        document.getElementById('route-form').addEventListener('submit', findRoute);
        document.getElementById('add-station-form').addEventListener('submit', addStation);
        document.getElementById('add-connection-form').addEventListener('submit', addConnection);
        document.getElementById('station-search').addEventListener('input', filterStations);
        document.getElementById('station-list-search').addEventListener('input', filterStationList);
        
        // Tab switching
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', function() {
                // Remove active class from all tabs and contents
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                
                // Add active class to clicked tab and corresponding content
                this.classList.add('active');
                document.getElementById(this.dataset.tab).classList.add('active');
            });
        });
    });
    
    // Initialize the map
    function initMap() {
        showLoading();
        
        // Load the map from the backend
        fetch('/get_map')
            .then(response => response.text())
            .then(html => {
                const mapContainer = document.getElementById('map');
                mapContainer.innerHTML = html;
                hideLoading();
            })
            .catch(error => {
                console.error('Error loading map:', error);
                hideLoading();
                showStatusMessage('Failed to load the map. Please refresh the page.', 'error');
            });
    }
    
    // Load stations from the backend
    function loadStations() {
        fetch('/get_stations')
            .then(response => response.json())
            .then(data => {
                stations = data;
                
                // Populate station dropdowns
                const srcSelect = document.getElementById('src-station');
                const dstSelect = document.getElementById('dst-station');
                const connSrcSelect = document.getElementById('conn-src');
                const connDstSelect = document.getElementById('conn-dst');
                const stationList = document.getElementById('station-list');
                
                srcSelect.innerHTML = '<option value="">Select Origin</option>';
                dstSelect.innerHTML = '<option value="">Select Destination</option>';
                connSrcSelect.innerHTML = '<option value="">Select Station</option>';
                connDstSelect.innerHTML = '<option value="">Select Station</option>';
                stationList.innerHTML = '';
                
                // Sort stations by name
                const sortedStations = Object.entries(stations).sort((a, b) => 
                    a[1].name.localeCompare(b[1].name)
                );
                
                for (const [code, data] of sortedStations) {
                    // Add to dropdowns
                    const option = document.createElement('option');
                    option.value = code;
                    option.textContent = `${data.name} (${code})`;
                    option.dataset.name = data.name.toLowerCase();
                    option.dataset.code = code.toLowerCase();
                    
                    srcSelect.appendChild(option.cloneNode(true));
                    dstSelect.appendChild(option.cloneNode(true));
                    connSrcSelect.appendChild(option.cloneNode(true));
                    connDstSelect.appendChild(option.cloneNode(true));
                    
                    // Add to station list
                    const stationItem = document.createElement('div');
                    stationItem.className = 'station-item';
                    stationItem.dataset.name = data.name.toLowerCase();
                    stationItem.dataset.code = code.toLowerCase();
                    
                    const stationInfo = document.createElement('div');
                    stationInfo.innerHTML = `
                        <strong>${data.name}</strong> (${code})
                        <div>Importance: ${data.importance || 5}/10</div>
                    `;
                    
                    const removeButton = document.createElement('button');
                    removeButton.type = 'button';
                    removeButton.innerHTML = '<i class="fas fa-trash"></i>';
                    removeButton.style.backgroundColor = 'transparent';
                    removeButton.style.color = '#dc3545';
                    removeButton.style.border = '1px solid #dc3545';
                    removeButton.style.padding = '5px 10px';
                    
                    removeButton.addEventListener('click', function(e) {
                        e.stopPropagation();
                        if (confirm(`Do you want to remove ${data.name} (${code})?`)) {
                            removeStation(code);
                        }
                    });
                    
                    stationItem.appendChild(stationInfo);
                    stationItem.appendChild(removeButton);
                    stationList.appendChild(stationItem);
                }
            })
            .catch(error => {
                console.error('Error loading stations:', error);
                showStatusMessage('Failed to load stations. Please refresh the page.', 'error');
            });
    }
    
    // Filter stations in dropdown based on search
    function filterStations() {
        const searchTerm = document.getElementById('station-search').value.toLowerCase();
        const srcSelect = document.getElementById('src-station');
        const dstSelect = document.getElementById('dst-station');
        
        Array.from(srcSelect.options).forEach(option => {
            if (option.value === '') return; // Skip the placeholder
            
            const name = option.dataset.name;
            const code = option.dataset.code;
            
            if (name && code) {
                const isVisible = name.includes(searchTerm) || code.includes(searchTerm);
                option.style.display = isVisible ? '' : 'none';
            }
        });
        
        Array.from(dstSelect.options).forEach(option => {
            if (option.value === '') return; // Skip the placeholder
            
            const name = option.dataset.name;
            const code = option.dataset.code;
            
            if (name && code) {
                const isVisible = name.includes(searchTerm) || code.includes(searchTerm);
                option.style.display = isVisible ? '' : 'none';
            }
        });
    }
    
    // Filter station list based on search
    function filterStationList() {
        const searchTerm = document.getElementById('station-list-search').value.toLowerCase();
        const stationItems = document.querySelectorAll('.station-item');
        
        stationItems.forEach(item => {
            const name = item.dataset.name;
            const code = item.dataset.code;
            
            if (name && code) {
                const isVisible = name.includes(searchTerm) || code.includes(searchTerm);
                item.style.display = isVisible ? '' : 'none';
            }
        });
    }
    
    // Find a route between two stations
    function findRoute(event) {
        event.preventDefault();
        
        const src = document.getElementById('src-station').value;
        const dst = document.getElementById('dst-station').value;
        const departureTime = document.getElementById('departure-time').value;
        const priority = document.getElementById('priority').value;
        
        if (!src || !dst || !departureTime) {
            showStatusMessage('Please select origin, destination, and departure time.', 'warning');
            return;
        }
        
        if (src === dst) {
            showStatusMessage('Origin and destination cannot be the same.', 'warning');
            return;
        }
        
        showLoading();
        
        const requestData = {
            src: src,
            dst: dst,
            departure_time: departureTime,
            priority: priority
        };
        
        fetch('/find_route', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        })
        .then(response => response.json())
        .then(data => {
            currentRoute = data;
            displayRouteDetails(data);
            updateMapWithRoute(data);
            hideLoading();
            showStatusMessage('Route found successfully!', 'success');
        })
        .catch(error => {
            hideLoading();
            console.error('Error finding route:', error);
            showStatusMessage('Error finding route. Please try again.', 'error');
        });
    }
    
    // Update map with the selected route
    function updateMapWithRoute(route) {
        fetch('/get_route_map', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ route: route })
        })
        .then(response => response.text())
        .then(html => {
            const mapContainer = document.getElementById('map');
            mapContainer.innerHTML = html;
        })
        .catch(error => {
            console.error('Error updating map:', error);
            showStatusMessage('Error updating map. Please try again.', 'error');
        });
    }
    
    // Display route details in the sidebar
    function displayRouteDetails(route) {
        const routeDetails = document.getElementById('route-details');
        const routeSummary = document.getElementById('route-summary');
        const routeSegments = document.getElementById('route-segments');
        const fareDetails = document.getElementById('fare-details');
        
        // Show the route details section
        routeDetails.style.display = 'block';
        
        // Display summary
        const delay = route.predicted_delay.toFixed(1);
        const delayClass = route.predicted_delay > 30 ? 'delay-warning' : '';
        
        routeSummary.innerHTML = `
            <h4><i class="fas fa-info-circle"></i> Journey Summary</h4>
            <p><strong>Total Distance:</strong> ${route.total_distance.toFixed(1)} km</p>
            <p><strong>Total Travel Time:</strong> ${route.total_time.toFixed(1)} hours</p>
            <p><strong>Predicted Delay:</strong> <span class="${delayClass}">${delay} minutes</span></p>
            <p><strong>Departure:</strong> ${formatDateTime(route.departure_time)}</p>
            <p><strong>Arrival:</strong> ${formatDateTime(route.arrival_time)}</p>
            <p><strong>Trains:</strong> ${route.trains.join(', ')}</p>
        `;
        
        // Display segments
        routeSegments.innerHTML = '<h4><i class="fas fa-exchange-alt"></i> Route Segments</h4>';
        
        route.route_segments.forEach((segment, index) => {
            const segmentDiv = document.createElement('div');
            segmentDiv.className = 'route-segment';
            segmentDiv.innerHTML = `
                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                    <i class="fas fa-train" style="margin-right: 8px; color: var(--accent-color);"></i>
                    <strong>${index + 1}. ${segment.from.name} (${segment.from.code}) → ${segment.to.name} (${segment.to.code})</strong>
                </div>
                <p>Distance: ${segment.distance.toFixed(1)} km</p>
                <p>Travel Time: ${segment.time.toFixed(1)} hours</p>
                <p>Train: ${segment.train}</p>
            `;
            routeSegments.appendChild(segmentDiv);
        });
        
        // Display fare details
        fareDetails.innerHTML = `
            <h4><i class="fas fa-ticket-alt"></i> Fare Details</h4>
            <div class="fare-option">
                <span class="fare-class">Sleeper Class:</span>
                <span class="fare-amount">₹${route.fare.sleeper}</span>
            </div>
            <div class="fare-option">
                <span class="fare-class">AC 3-Tier:</span>
                <span class="fare-amount">₹${route.fare.ac_three_tier}</span>
            </div>
            <div class="fare-option">
                <span class="fare-class">AC 2-Tier:</span>
                <span class="fare-amount">₹${route.fare.ac_two_tier}</span>
            </div>
            <div class="fare-option">
                <span class="fare-class">AC First Class:</span>
                <span class="fare-amount">₹${route.fare.ac_first_class}</span>
            </div>
        `;
        
        // Scroll to route details
        routeDetails.scrollIntoView({ behavior: 'smooth' });
    }
    
    // Format date and time for display
    function formatDateTime(dateTimeStr) {
        const options = { 
            year: 'numeric', 
            month: 'short', 
            day: 'numeric', 
            hour: '2-digit', 
            minute: '2-digit'
        };
        return new Date(dateTimeStr).toLocaleString('en-IN', options);
    }
    
    // Add a new station
    function addStation(event) {
        event.preventDefault();
        
        const code = document.getElementById('station-code').value.toUpperCase();
        const name = document.getElementById('station-name').value;
        const lat = parseFloat(document.getElementById('station-lat').value);
        const lon = parseFloat(document.getElementById('station-lon').value);
        const importance = parseInt(document.getElementById('station-importance').value);
        
        if (!code || !name || isNaN(lat) || isNaN(lon)) {
            showStatusMessage('Please fill in all fields with valid values.', 'warning');
            return;
        }
        
        showLoading();
        
        const requestData = {
            action: 'add_station',
            code: code,
            name: name,
            lat: lat,
            lon: lon,
            importance: importance
        };
        
        fetch('/update_network', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        })
        .then(response => response.json())
        .then(data => {
            hideLoading();
            
            if (data.error) {
                showStatusMessage(data.error, 'error');
            } else {
                showStatusMessage(`Station ${name} (${code}) added successfully.`, 'success');
                // Reset form
                document.getElementById('add-station-form').reset();
                // Reload stations
                loadStations();
                // Reload map
                initMap();
            }
        })
        .catch(error => {
            hideLoading();
            console.error('Error adding station:', error);
            showStatusMessage('Failed to add station. Please try again.', 'error');
        });
    }
    
    // Remove a station
    function removeStation(code) {
        showLoading();
        
        const requestData = {
            action: 'remove_station',
            code: code
        };
        
        fetch('/update_network', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        })
        .then(response => response.json())
        .then(data => {
            hideLoading();
            
            if (data.error) {
                showStatusMessage(data.error, 'error');
            } else {
                showStatusMessage(`Station ${code} removed successfully.`, 'success');
                // Reload stations
                loadStations();
                // Reload map
                initMap();
            }
        })
        .catch(error => {
            hideLoading();
            console.error('Error removing station:', error);
            showStatusMessage('Failed to remove station. Please try again.', 'error');
        });
    }
    
    // Add a new connection
    function addConnection(event) {
        event.preventDefault();
        
        const src = document.getElementById('conn-src').value;
        const dst = document.getElementById('conn-dst').value;
        const distance = parseFloat(document.getElementById('conn-distance').value);
        const time = parseFloat(document.getElementById('conn-time').value);
        const train = document.getElementById('conn-train').value;
        
        if (!src || !dst || isNaN(distance) || isNaN(time) || !train) {
            showStatusMessage('Please fill in all fields with valid values.', 'warning');
            return;
        }
        
        if (src === dst) {
            showStatusMessage('Source and destination stations cannot be the same.', 'warning');
            return;
        }
        
        showLoading();
        
        const requestData = {
            action: 'add_connection',
            src: src,
            dst: dst,
            distance: distance,
            time: time,
            train: train
        };
        
        fetch('/update_network', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        })
        .then(response => response.json())
        .then(data => {
            hideLoading();
            
            if (data.error) {
                showStatusMessage(data.error, 'error');
            } else {
                showStatusMessage(`Connection between ${src} and ${dst} added successfully.`, 'success');
                // Reset form
                document.getElementById('add-connection-form').reset();
                // Reload map
                initMap();
            }
        })
        .catch(error => {
            hideLoading();
            console.error('Error adding connection:', error);
            showStatusMessage('Failed to add connection. Please try again.', 'error');
        });
    }
    
    // Show loading spinner
    function showLoading() {
        document.getElementById('map-loading').style.display = 'flex';
    }
    
    // Hide loading spinner
    function hideLoading() {
        document.getElementById('map-loading').style.display = 'none';
    }
    
    // Show status message
    function showStatusMessage(message, type) {
        const statusElement = document.getElementById('status-message');
        statusElement.textContent = message;
        statusElement.className = 'status-message';
        statusElement.classList.add(`status-${type}`);
        statusElement.style.display = 'block';
        
        // Hide after 5 seconds
        setTimeout(() => {
            statusElement.style.display = 'none';
        }, 5000);
    }
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
    
    # Use cached route if available
    try:
        cached_route = get_cached_route(src, dst, departure_str, priority)
        return jsonify(cached_route)
    except Exception as e:
        print(f"Error in find_route: {e}")
        # If there's an error, create a fallback route
        departure_time = datetime.strptime(departure_str, '%Y-%m-%dT%H:%M')
        
        # Calculate straight-line distance
        lat1, lon1 = STATIONS[src]["lat"], STATIONS[src]["lon"]
        lat2, lon2 = STATIONS[dst]["lat"], STATIONS[dst]["lon"]
        
        # Simple distance calculation
        dist = ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5 * 111  # Approx km per degree
        time = dist / 60  # Assume 60 km/h average speed
        
        # Create a fallback route
        fallback_route = {
            'path': [src, dst],
            'route_segments': [{
                'from': {
                    'code': src,
                    'name': STATIONS[src]['name'],
                    'lat': STATIONS[src]['lat'],
                    'lon': STATIONS[src]['lon']
                },
                'to': {
                    'code': dst,
                    'name': STATIONS[dst]['name'],
                    'lat': STATIONS[dst]['lat'],
                    'lon': STATIONS[dst]['lon']
                },
                'distance': dist,
                'time': time,
                'train': "Direct Express"
            }],
            'total_distance': dist,
            'total_time': time,
            'predicted_delay': 30,
            'departure_time': departure_time.strftime('%Y-%m-%d %H:%M'),
            'arrival_time': (departure_time + timedelta(hours=time, minutes=30)).strftime('%Y-%m-%d %H:%M'),
            'trains': ["Direct Express"],
            'fare': {
                'sleeper': round(dist * 1.5),
                'ac_three_tier': round(dist * 1.5 * 2.5),
                'ac_two_tier': round(dist * 1.5 * 2.5 * 1.4),
                'ac_first_class': round(dist * 1.5 * 2.5 * 2)
            }
        }
        
        return jsonify(fallback_route)

@app.route('/get_map')
def get_map():
    # Create a basic map with all stations and connections
    m = create_railway_map()
    return m._repr_html_()

@app.route('/get_route_map', methods=['POST'])
def get_route_map():
    data = request.get_json()
    route = data.get('route')
    
    # Create a map highlighting the selected route
    m = create_railway_map(route)
    return m._repr_html_()

@app.route('/get_stations')
def get_stations():
    return jsonify(STATIONS)

@app.route('/update_network', methods=['POST'])
def update_network():
    global STATIONS, CONNECTIONS, railway_graph
    
    data = request.get_json()
    action = data.get('action')
    
    if action == 'add_station':
        code = data.get('code')
        name = data.get('name')
        lat = data.get('lat')
        lon = data.get('lon')
        importance = data.get('importance', 5)
        
        if code in STATIONS:
            return jsonify({'error': 'Station code already exists'}), 400
        
        STATIONS[code] = {
            'name': name,
            'lat': lat,
            'lon': lon,
            'importance': importance
        }
        
    elif action == 'remove_station':
        code = data.get('code')
        
        if code not in STATIONS:
            return jsonify({'error': 'Station not found'}), 404
        
        # Remove the station and all its connections
        del STATIONS[code]
        CONNECTIONS = [c for c in CONNECTIONS if c[0] != code and c[1] != code]
        
    elif action == 'add_connection':
        src = data.get('src')
        dst = data.get('dst')
        distance = data.get('distance')
        time = data.get('time')
        train = data.get('train', 'Express')
        
        if src not in STATIONS or dst not in STATIONS:
            return jsonify({'error': 'One or both stations not found'}), 404
        
        # Check if connection already exists
        for i, conn in enumerate(CONNECTIONS):
            if (conn[0] == src and conn[1] == dst) or (conn[0] == dst and conn[1] == src):
                CONNECTIONS[i] = (src, dst, distance, time, train)
                break
        else:
            CONNECTIONS.append((src, dst, distance, time, train))
    
    elif action == 'remove_connection':
        src = data.get('src')
        dst = data.get('dst')
        
        CONNECTIONS = [c for c in CONNECTIONS if not ((c[0] == src and c[1] == dst) or (c[0] == dst and c[1] == src))]
    
    # Recreate the graph with updated data
    railway_graph = create_railway_graph()
    
    # Clear the route cache when the network changes
    get_cached_route.cache_clear()
    
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True)