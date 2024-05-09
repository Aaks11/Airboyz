import numpy as np
import matplotlib.pyplot as plt

# Grid setup
grid_size = 50
x = np.linspace(-grid_size / 2, grid_size / 2, grid_size)
y = np.linspace(-grid_size / 2, grid_size / 2, grid_size)
xx, yy = np.meshgrid(x, y)

# Flatten coordinates
coordinates = np.vstack([xx.ravel(), yy.ravel()]).T

# Random seed
np.random.seed(0)

# Waypoints
num_way_points = coordinates.shape[0] // 10
way_points_indices = np.random.choice(coordinates.shape[0], num_way_points, replace=False)
way_points = coordinates[way_points_indices]

# Remaining coordinates
remaining_coordinates = np.delete(coordinates, way_points_indices, axis=0)

# Airport locations
num_airport_locations = remaining_coordinates.shape[0] // 40
airport_locations_indices = np.random.choice(remaining_coordinates.shape[0], num_airport_locations, replace=False)
airport_locations = remaining_coordinates[airport_locations_indices]

# Airports
toff_aport = airport_locations[0]
land_aport = airport_locations[51]

# Distances
per_distances = np.abs(np.cross(land_aport - toff_aport, toff_aport - way_points)) / np.linalg.norm(
    land_aport - toff_aport)

# Proximity
proxy_margin = 1.0
close_way_points = way_points[per_distances <= proxy_margin]

# Vector from takeoff to landing
takeoff_to_landing = land_aport - toff_aport

# Norm of vector
takeoff_to_landing_norm = np.linalg.norm(takeoff_to_landing)

# Vectors to waypoints
vectors_to_waypoints = close_way_points - toff_aport

# Norms of vectors
norms_to_waypoints = np.linalg.norm(vectors_to_waypoints, axis=1)

# Mask for waypoints
boolean_mask = norms_to_waypoints <= (takeoff_to_landing_norm + 0)

# Optimized waypoints
optimized_way_points = close_way_points[boolean_mask]

print(f"number of optimized waypoints: {len(optimized_way_points)}")
print('number of waypoints', len(way_points))
# Path creation
path = [toff_aport]  # Start with the takeoff location.

# Waypoint addition with angle and distance consideration
while len(optimized_way_points) > 0:
    if len(path) > 1:
        current_direction = path[-1] - path[-2]
    else:
        current_direction = land_aport - toff_aport

    min_cost = float('inf')
    next_waypoint = None
    next_index = -1

    # Iterate through waypoints to find the optimal next step
    for index, waypoint in enumerate(optimized_way_points):
        travel_vector = waypoint - path[-1]
        travel_distance = np.linalg.norm(travel_vector)
        total_cost = travel_distance  # Could include other factors later
        print("total_cost", total_cost)

        # Check direction
        cosine_angle = np.dot(current_direction, travel_vector) / (np.linalg.norm(current_direction) * np.linalg.norm(travel_vector) + np.finfo(float).eps)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        print("angle", angle)

        # Angle and cost-based selection
        if np.abs(angle) <= np.radians(40) and total_cost < min_cost:
            print("total_cost", total_cost)
            print("min cost", min_cost)
            min_cost = total_cost
            print("new min cost", min_cost)
            next_waypoint = waypoint
            next_index = index

    # Add the best waypoint found to the path or break if none found
    if next_waypoint is not None:
        path.append(next_waypoint)
        optimized_way_points = np.delete(optimized_way_points, next_index, axis=0)
    else:
        print("No suitable next waypoint found that meets the angle and distance constraints.")
        break

# Landing airport addition
path.append(land_aport)
path = np.array(path)

# Plotting
plt.figure(figsize=(6, 6))
plt.scatter(coordinates[:, 0], coordinates[:, 1], color='gray', s=10, label='Grid')
plt.scatter(way_points[:, 0], way_points[:, 1], color='blue', s=50, label='Way Points')
plt.scatter(airport_locations[:, 0], airport_locations[:, 1], color='red', s=50, label='Airport Locations')
plt.scatter(toff_aport[0], toff_aport[1], color='yellow', edgecolor='black', s=100, label='Takeoff Airport')
plt.scatter(land_aport[0], land_aport[1], color='purple', edgecolor='black', s=100, label='Landing Airport')
plt.plot(path[:, 0], path[:, 1], color='green', label='Path')
plt.title('Way Points and Airport Locations on the Grid')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
