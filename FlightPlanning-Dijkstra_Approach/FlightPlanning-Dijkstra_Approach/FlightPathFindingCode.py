
################################################################################
#                                                                              #
#                          Flight Planning Code                                #
#                    using Dijkstra's Approach                                 #
#                                                                              #
# Author: [Luca Mattiocco]                                                     #
# Date: April 2024                                                             #
# Affiliation: Cranfield University                                            #
#                                                                              #
# Description:                                                                 #
# This code implements a flight planning algorithm using Dijkstra's approach   #
# to find the shortest path between two airports based on given flight         #
# connections and distances.                                                   #
#                                                                              #
#                                                                              #
#                                                                              #
################################################################################


import folium
import pandas as pd
import numpy as np
import random
import heapq
from scipy.spatial import Delaunay
from sklearn.cluster import KMeans

# Read CSV files
takeoff_df = pd.read_csv('Takeoff_info.csv')
landing_df = pd.read_csv('Landing_info.csv')

# Extract latitude and longitude for takeoff and landing points
takeoff_lat, takeoff_lon = takeoff_df.iloc[0]['Latitude'], takeoff_df.iloc[0]['Longitude']
landing_lat, landing_lon = landing_df.iloc[0]['Latitude'], landing_df.iloc[0]['Longitude']

# Create a map centered around the mean latitude and longitude of all points
mean_lat = (takeoff_lat + landing_lat) / 2
mean_lon = (takeoff_lon + landing_lon) / 2
mymap = folium.Map(location=[mean_lat, mean_lon], zoom_start=10)

# Add markers for takeoff and landing points
folium.Marker([takeoff_lat, takeoff_lon], popup='Takeoff Point', icon=folium.Icon(color='blue')).add_to(mymap)
folium.Marker([landing_lat, landing_lon], popup='Landing Point', icon=folium.Icon(color='red')).add_to(mymap)

# List of file paths and corresponding colors
file_paths = {
    "Reykjavik_FIR_Waypoints.csv": "blue",
    "Ireland_FIR_Waypoints.csv": "green",
    "UK_FIR_Waypoints.csv": "orange"
}

# Function to add markers to the map
def add_waypoint_markers(map, waypoints, waypoint_names, color):
    for waypoint, name in zip(waypoints, waypoint_names):
        folium.Marker([waypoint[0], waypoint[1]], popup=name, icon=folium.Icon(color=color)).add_to(map)

# Function to find the shortest path between start and end points using Dijkstra's algorithm
def dijkstra(graph, start, end):
    # Priority queue to store vertices and their distances
    pq = [(0, start, [start])]
    # Set to store visited vertices
    visited = set()
    # Variable to store the shortest path
    shortest_path = []
    while pq:
        current_distance, current_vertex, path = heapq.heappop(pq)
        # If the current vertex is the end point, return the path
        if current_vertex == end:
            shortest_path = path
            break
        # If the current vertex has already been visited, skip it
        if current_vertex in visited:
            continue
        # Mark the current vertex as visited
        visited.add(current_vertex)
        # Explore neighbors of the current vertex
        for neighbor, weight in graph[current_vertex].items():
            if neighbor not in visited:
                # Calculate the distance from start to neighbor through current vertex
                distance = current_distance + weight
                # Add neighbor to the priority queue with updated distance and path
                heapq.heappush(pq, (distance, neighbor, path + [neighbor]))
    return shortest_path

# Function to calculate distance between two points with added weights
def weighted_distance(point1, point2, weight):
    return np.linalg.norm(np.array(point1) - np.array(point2)) + weight

# Store all waypoints and their names
all_waypoints = []
all_waypoint_names = []

# Load and process data for each file
for file_path, color in file_paths.items():
    data = pd.read_csv(file_path, delimiter=';')
    # Remove extra spaces around column names
    data.columns = data.columns.str.strip()
    # Store waypoints and names
    waypoints = np.array(data[['Latitude', 'Longitude']])
    waypoint_names = np.array(data['Point'])
    all_waypoints.extend(waypoints)
    all_waypoint_names.extend(waypoint_names)

# Convert waypoints and names to numpy arrays
all_waypoints = np.array(all_waypoints)
all_waypoint_names = np.array(all_waypoint_names)

# Clustering des waypoints avec K-means
num_clusters = 200  # Nombre de clusters à former
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(all_waypoints)
cluster_centers = kmeans.cluster_centers_

# Find nearest waypoints for each cluster centroid
def find_nearest_waypoint(cluster_centers, waypoints, waypoint_names):
    nearest_waypoints = []
    nearest_waypoint_names = []
    for centroid in cluster_centers:
        nearest_index = np.argmin(np.linalg.norm(waypoints - centroid, axis=1))
        nearest_waypoint = waypoints[nearest_index]
        nearest_waypoints.append(nearest_waypoint)
        nearest_waypoint_name = waypoint_names[nearest_index]
        nearest_waypoint_names.append(nearest_waypoint_name)
    return np.array(nearest_waypoints), np.array(nearest_waypoint_names)

# Plot nearest waypoints instead of cluster centroids
def plot_waypoints(map, waypoints, waypoint_names, color):
    add_waypoint_markers(map, waypoints, waypoint_names, color)

# Plot waypoints on the map
nearest_waypoints, nearest_waypoint_names = find_nearest_waypoint(cluster_centers, all_waypoints, all_waypoint_names)
plot_waypoints(mymap, nearest_waypoints, nearest_waypoint_names, 'red')

# Choose random starting and ending points
use_random_weights = input("Use weighted connection? (Yes/No): ")
if use_random_weights.lower() == "yes":
    weight_connection = random.uniform(1, 50)
    weight_base = 1  # Assign fixed weight for base points
else:
    weight_connection = 1
    weight_base = 1  # Assign fixed weight for base points

# Replace starting point with takeoff point coordinates
takeoff_point = [takeoff_lat, takeoff_lon]

# Replace ending point with landing point coordinates
landing_point = [landing_lat, landing_lon]

# Plot starting and ending points
folium.Marker([takeoff_point[0], takeoff_point[1]], popup="Takeoff Point", icon=folium.Icon(color='green')).add_to(mymap)
folium.Marker([landing_point[0], landing_point[1]], popup="Landing Point", icon=folium.Icon(color='blue')).add_to(mymap)

# Calculate Delaunay triangulation for nearest waypoints including start and end points
all_points = np.vstack((nearest_waypoints, takeoff_point, landing_point))
tri = Delaunay(all_points)

# Construct graph from Delaunay triangulation for Dijkstra's algorithm with added weights
graph = {}

for indices in tri.simplices:
    for i in range(3):
        point1 = tuple(all_points[indices[i]])
        for j in range(i+1, 3):
            point2 = tuple(all_points[indices[j]])
            if point1 not in graph:
                graph[point1] = {}
            if point2 not in graph:
                graph[point2] = {}
            if use_random_weights.lower() == "yes":
                weight = random.uniform(1, 50)
            else:
                weight = 1
            # Assign different weights for connections based on user's choice
            graph[point1][point2] = weighted_distance(point1, point2, weight)
            graph[point2][point1] = weighted_distance(point1, point2, weight)

# Find shortest path using Dijkstra's algorithm
shortest_path = dijkstra(graph, tuple(takeoff_point), tuple(landing_point))

# Initialize set to store visited waypoints
visited_waypoints = set()

# Add waypoints along the shortest path to visited waypoints
for point in shortest_path:
    visited_waypoints.add(tuple(point))

# Initialize list to store visited waypoints
visited_waypoints_list = []

# Create a list of visited waypoints in the correct order
for waypoint_name in nearest_waypoint_names:
    if tuple(nearest_waypoints[np.where(nearest_waypoint_names == waypoint_name)][0]) in visited_waypoints:
        visited_waypoints_list.append(waypoint_name)

# Print the list of visited waypoints
print("Visited Waypoints:")
for waypoint_name in visited_waypoints_list:
    print(waypoint_name)

# Plot visited waypoints on the map
for waypoint, name in zip(nearest_waypoints, nearest_waypoint_names):
    if tuple(waypoint) in visited_waypoints:
        folium.Marker([waypoint[0], waypoint[1]], popup=name, icon=folium.Icon(color='purple')).add_to(mymap)
    else:
        folium.Marker([waypoint[0], waypoint[1]], popup=name, icon=folium.Icon(color='red')).add_to(mymap)

# Plot shortest path on the map
folium.PolyLine(shortest_path, color='green').add_to(mymap)

# Save the map with visited waypoints, shortest path, and original waypoints
mymap.save("Waypoints_Cluster_Connections_Delaunay_with_names_and_shortest_path_and_visited.html")
