######################################################################################
#                                                                                    #
#                            Network Mesh Visualization                              #
#                                                                                    #
# Author: [Luca Mattiocco]                                                           #
# Date: April 2024                                                                   #
# Affiliation: Cranfield University                                                  #
#                                                                                    #
# Description:                                                                       #
# This code generates a random mesh of points and calculates the shortest path       #
# using Dijkstra's algorithm. It then modifies the mesh with a corridor and          #
# recalculates the shortest path. Finally, it plots the original and modified meshes #
# along with their respective shortest paths.                                        #
#                                                                                    #
#                                                                                    #
#                                                                                    #
######################################################################################

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial import Delaunay
from shapely.geometry import LineString

# Function to randomly generate edge weights
def generate_edge_weights(num_edges):
    return np.random.randint(2, 31, num_edges)

# Function to map weight values to colors
def map_weight_to_color(weight):
    if weight <= 5:
        return 'blue'
    elif weight <= 10:
        return 'green'
    elif weight <= 15:
        return 'yellow'
    elif weight <= 20:
        return 'orange'
    elif weight <= 25:
        return 'red'
    else:
        return 'purple'

# Function to generate random points in a given space
def generate_points(num_points, xlim, ylim):
    return np.random.uniform(xlim[0], xlim[1], num_points), np.random.uniform(ylim[0], ylim[1], num_points)

# Number of random points
num_points = 100

# Limits of the space where random points are generated
xlim = (0, 10)
ylim = (0, 10)

# Generate random points
x, y = generate_points(num_points, xlim, ylim)

# Define start and end points
start_point = (1, 1)
end_point = (9, 9)

# Add start and end points to the list of points
x = np.append(x, [start_point[0], end_point[0]])
y = np.append(y, [start_point[1], end_point[1]])

# Create mesh using Delaunay algorithm
points = np.column_stack((x, y))
tri = Delaunay(points)

# Randomly generate edge values
num_edges = len(tri.simplices)
edge_weights = generate_edge_weights(num_edges)

# Create a graph
G = nx.Graph()

# Add nodes to the graph
for i in range(len(points)):
    G.add_node(i, pos=points[i])

# Add edges to the graph with weights
for i, simplex in enumerate(tri.simplices):
    weight = edge_weights[i]
    G.add_edge(simplex[0], simplex[1], weight=weight)
    G.add_edge(simplex[1], simplex[2], weight=weight)
    G.add_edge(simplex[2], simplex[0], weight=weight)

# Calculate shortest path using Dijkstra's algorithm for the original mesh
shortest_path_original = nx.shortest_path(G, source=len(points)-2, target=len(points)-1, weight='weight')

# Create corridor between two parallel lines
directing_vector = np.array(end_point) - np.array(start_point)
corridor_width = 0.075
corridor_line1 = LineString([(start_point[0] - directing_vector[1]*corridor_width, start_point[1] + directing_vector[0]*corridor_width),
                             (end_point[0] - directing_vector[1]*corridor_width, end_point[1] + directing_vector[0]*corridor_width)])
corridor_line2 = LineString([(start_point[0] + directing_vector[1]*corridor_width, start_point[1] - directing_vector[0]*corridor_width),
                             (end_point[0] + directing_vector[1]*corridor_width, end_point[1] - directing_vector[0]*corridor_width)])

# Update edge weights inside the corridor for the modified mesh
G_modified = G.copy()
for u, v, weight in G_modified.edges(data='weight'):
    if LineString([points[u], points[v]]).intersects(corridor_line1) or LineString([points[u], points[v]]).intersects(corridor_line2):
        G_modified[u][v]['weight'] -= 7
        G_modified[u][v]['weight'] = max(G_modified[u][v]['weight'], 1)

# Calculate shortest path using Dijkstra's algorithm for the modified mesh
shortest_path_modified = nx.shortest_path(G_modified, source=len(points)-2, target=len(points)-1, weight='weight')

# Plot the original and modified meshes with their respective shortest paths
plt.figure(figsize=(12, 5))

# Plot the original mesh
plt.subplot(1, 2, 1)
plt.triplot(x, y, tri.simplices, color='gray')  # Mesh in gray
plt.plot(x, y, 'o', color='black')  # Points in black
plt.plot(start_point[0], start_point[1], 'o', color='blue')
plt.plot(end_point[0], end_point[1], 'ro')
plt.title('Original Mesh')

# Plot the shortest path on the original mesh
for i in range(len(shortest_path_original)-1):
    plt.plot([points[shortest_path_original[i]][0], points[shortest_path_original[i+1]][0]], 
             [points[shortest_path_original[i]][1], points[shortest_path_original[i+1]][1]], 
             color='cyan', linewidth=2)

# Plot the modified mesh
plt.subplot(1, 2, 2)
plt.triplot(x, y, tri.simplices, color='gray')  # Mesh in gray
plt.plot(x, y, 'o', color='black')  # Points in black
plt.plot(start_point[0], start_point[1], 'o', color='blue')
plt.plot(end_point[0], end_point[1], 'ro')
plt.title('Modified Mesh')

# Plot the shortest path on the modified mesh
for i in range(len(shortest_path_modified)-1):
    plt.plot([points[shortest_path_modified[i]][0], points[shortest_path_modified[i+1]][0]], 
             [points[shortest_path_modified[i]][1], points[shortest_path_modified[i+1]][1]], 
             color='cyan', linewidth=2)

plt.tight_layout()
plt.show()
