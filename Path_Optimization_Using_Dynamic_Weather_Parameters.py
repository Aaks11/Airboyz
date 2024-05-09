import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon
from datetime import datetime, timedelta
from matplotlib.patches import Circle

   #####################################################
   # Code Developed by                                 #
   # Swapnil Patil                                     #
   # Aerospace Computational Engineering MSc 2023-2024 #
   # Cranfield University                              #
   # swapnil.patil.897@cranfield.ac.uk                 #
   #                                                   #
   #####################################################

#############################################################################################################
# This script provides a foundation for path optimization using dynamic weather.                            #
# It demonstrates a complex integration of weather pattern simulations, aircraft path                       #
# calculations, and real-time graphical updates, setting a solid groundwork for further development in path #
# optimization technologies.                                                                                #
#############################################################################################################

# Key Areas for Future Development:
# - Precipitation Forecasting: Currently, the implementation considers precipitation
#   only at specific waypoints and not along the path segments connecting these points.
#   Extending the precipitation evaluation to these segments could provide a more
#   comprehensive understanding of environmental conditions affecting the entire flight
#   route.
# - Temporal Coherence: There are discrepancies in how temporal elements are handled
#   between the simulation of precipitation and aircraft movements. Aligning the
#   temporal logic used for both will be crucial for ensuring the forecasts are accurate
#   and reliable.
# - Path and Waypoint Optimization: This script lays the groundwork for path optimization
#   under dynamic environmental conditions. It employs multiple iterations and recalculations,
#   which could be made more efficient through caching results or precomputing reusable
#   values. Simplifying complex conditions and modularizing the code into smaller, more
#   manageable functions would also enhance maintainability and performance.
#
# # =============================================================================
# # This foundation invites future contributions that could refine and expand upon the initial
# # models, particularly focusing on implementing the more comprehensive cost functions
# # and optimizing real-time decision-making for aircraft navigation. By improving these aspects,
# # the simulation will become a more powerful tool for predictive analytics and operational
# # planning in adverse weather conditions. Enhancements in these areas will enable more precise
# # predictions and efficient navigational strategies, thereby increasing safety and efficiency
# # in flight operations under challenging weather scenarios.
# =============================================================================


# Threshold for high precipitation (adjust as needed)
precipitation_threshold = 2
# This is an example value

# Dictionary to store high precipitation events
high_precipitation_events = {}

# Dictionary to store timestamps and positions of aircraft
aircraft_waypoints_timestamps = {}


# Initialize global variables
global_min_precipitation = float('inf')
global_max_precipitation = float('-inf')

np.random.seed(42)  # Setting a fixed seed for reproducibility


def scale_precipitation(value, min_val, max_val, new_min=1, new_max=20):
    # Values of precipitation are scaled between 1 and 20 for convenience
    return new_min + (value - min_val) * (new_max - new_min) / (max_val - min_val)


def generate_wind_data(N):
    """
        Generate a simulation of wind vector components based on a potential field. This function
        simulates the wind over a unit square area by solving a discrete approximation of Laplace's equation
        to find a potential field, then calculates the gradient of this field to obtain the wind vectors.

        Args:
            N (int): The resolution of the grid. A higher value results in a finer grid.

        Returns:
            tuple: Two numpy arrays (Vx, Vy) representing the x and y components of the wind velocity at each grid point.
        """
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)
    phi = np.zeros((N, N))
    phi[0, :] = 143
    phi[-1, :] = 21
    phi[:, 0] = 49
    phi[:, -1] = 93
    for _ in range(5000):
        phi[1:-1, 1:-1] = (phi[:-2, 1:-1] + phi[2:, 1:-1] + phi[1:-1, :-2] + phi[1:-1, 2:]) / 4
    Vx, Vy = np.gradient(-phi)
    return Vx, Vy


def record_aircraft_timestamps(position, timestamp):
    """
    Record the timestamp and position of the aircraft.
    Args:
    position (np.array): The current position of the aircraft.
    timestamp (datetime): The current timestamp.

     Note:
        This function modifies a global dictionary named `aircraft_waypoints_timestamps` to store the timestamps.
    """
    # Round position to nearest integer for simplicity in storage
    position_tuple = tuple(np.round(position).astype(int))
    if position_tuple not in aircraft_waypoints_timestamps:
        aircraft_waypoints_timestamps[position_tuple] = timestamp
        print(f"Recorded waypoint at {position_tuple} on {timestamp}")


def calculate_travel_times(path, speed_kmph):
    """
    Calculate estimated travel times to each waypoint in the path.
    Args:
    path (np.array): Array of waypoints.
    speed_kmph (float): Aircraft speed in kilometers per hour.

    Returns:
    List of datetime objects representing the estimated times of arrival at each waypoint.
    """
    times = [datetime.now()]  # Start time is now
    total_distance = 0  # Total distance traveled
    for i in range(1, len(path)):
        distance = np.linalg.norm(path[i] - path[i - 1])
        total_distance += distance
        # Convert distance to travel time (distance in km / speed in km/h = time in hours)
        travel_time_hours = distance / speed_kmph
        # Convert hours to timedelta and add to the last time
        times.append(times[-1] + timedelta(hours=travel_time_hours))
    return times


def forecast_precipitation_along_path(path, travel_times, coordinates, grid_size, cov_scale, Vx, Vy, precipitation_sources):
    """
    Forecast precipitation at each waypoint at the expected arrival time.
    """
    forecasts = {}
    current_time = datetime.now()
    Vx_mean, Vy_mean = Vx.mean(), Vy.mean()

    for position, arrival_time in zip(path, travel_times):
        # Simulate the precipitation for a future time based on current data and weather patterns
        # Here, we assume the weather pattern changes slightly over time, represented by a Gaussian function with continuous input from cosine function as BC's
        # We may need a more complex model based on actual weather forecast data
        # Calculate the frame index based on the difference between the arrival time and current time
        frame = int((arrival_time - datetime.now()).total_seconds() / 60)  # Convert future time to frame index
        wind_vector = calculate_wind_vector(frame, Vx_mean, Vy_mean)

        # Simulate how the sources and precipitation might evolve by that frame
        # This is a simplistic assumption and may need a more detailed weather model
        timestamp = datetime.now() + timedelta(minutes=frame)
        centers = [source["position"] + wind_vector * frame for source in precipitation_sources.values()]
        precipitation_field = simulate_precipitation(coordinates, cov_scale, timestamp, precipitation_sources)

        # Find the precipitation value at the waypoint's grid position
        grid_x, grid_y = int(position[0]), int(position[1])
        if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
            forecasts[(grid_x, grid_y)] = (arrival_time, precipitation_field[grid_y, grid_x])

    return forecasts


def simulate_precipitation(coordinates, cov_scale, timestamp, precipitation_sources):
    """
        Simulates the precipitation at each point on a grid given multiple moving precipitation sources.

        Args:
            coordinates (np.array): A 2D array of x, y coordinates for grid points.
            cov_scale (float): Scale factor for the covariance of the precipitation distribution.
            timestamp (datetime): The current time for which precipitation is being simulated.
            precipitation_sources (dict): A dictionary of precipitation sources with their current positions and movement directions.

        Globals:
            global_min_precipitation (float): Tracks the minimum precipitation recorded globally.
            global_max_precipitation (float): Tracks the maximum precipitation recorded globally.

        Returns:
            np.array: A grid of scaled precipitation values.
        """
    global global_min_precipitation, global_max_precipitation
    total_precipitation = np.zeros((grid_size, grid_size))
    reshaped_coordinates = coordinates.reshape(grid_size * grid_size, 2)

    for name, source in precipitation_sources.items():
        center = source["position"]
        direction = source["direction"]

        # Update position with direction
        new_position = center + direction
        # Ensure the new position does not go out of the plot boundaries
        new_position = np.clip(new_position, 0, grid_size - 1)
        source["position"] = new_position

        covariance = cov_scale * np.eye(2)
        rv = multivariate_normal(mean=center, cov=covariance)
        precipitation_at_points = rv.pdf(reshaped_coordinates).reshape(grid_size, grid_size)
        total_precipitation += precipitation_at_points

        # Optionally move the source based on its direction
        # source["position"] += direction  # Update position based on the direction for dynamic movement

        # Update global precipitation min and max
    current_min = np.min(total_precipitation)
    current_max = np.max(total_precipitation)
    if current_min < global_min_precipitation:
        global_min_precipitation = current_min
    if current_max > global_max_precipitation:
        global_max_precipitation = current_max

        # Apply scaling to normalize precipitation data
    scaled_precipitation = scale_precipitation(total_precipitation, global_min_precipitation, global_max_precipitation)

    # Check for high precipitation and store the data
    high_precip = scaled_precipitation >= precipitation_threshold
    if np.any(high_precip):
        high_coords = np.argwhere(high_precip)
        for coord in high_coords:
            x, y = coord
            high_precipitation_events[(x, y, timestamp)] = scaled_precipitation[x, y]

    print(f'Min precipitation: {np.min(scaled_precipitation)}, Max precipitation: {np.max(scaled_precipitation)}')

    return scaled_precipitation


# Function to calculate the interpolated position and direction of the aircraft
def get_aircraft_position_and_angle(path, frame, total_frames):
    """
       Calculates the aircraft's position and direction angle along a defined path at a specified frame
       based on linear interpolation between path waypoints.

       Args:
           path (list or np.ndarray): A list of waypoints defining the path of the aircraft. Each waypoint is a tuple or list of length 2.
           frame (int): The current frame number to determine the position on the path.
           total_frames (int): The total number of frames in the animation or simulation.

       Returns:
           tuple: The interpolated position (np.ndarray) and the normalized direction vector (np.ndarray) of the aircraft.
       """
    for i, p in enumerate(path):
        if not (isinstance(p, (list, tuple, np.ndarray)) and len(p) == 2):
            print(f"Invalid element at index {i}: {p}")
    # Check that all waypoints are consistent and of length 2
    if not all(isinstance(p, (list, tuple, np.ndarray)) and len(p) == 2 for p in path):
        raise ValueError("All elements in 'path' must be sequences of length 2")

    # Convert path to numpy array if it's not already one
    path = np.array(path) if not isinstance(path, np.ndarray) else path

    # Ensure the path is a 2D array with shape (n, 2)
    if len(path) > 1 and path.ndim == 2 and path.shape[1] == 2:
        total_path_length = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
        travel_distance = (frame / total_frames) * total_path_length
        accumulated_distance = 0

        # Loop over the path to find the interpolated position for the current frame
        for i in range(1, len(path)):
            segment_length = np.linalg.norm(path[i] - path[i - 1])
            if accumulated_distance + segment_length >= travel_distance:
                ratio = (travel_distance - accumulated_distance) / segment_length
                position = path[i - 1] + ratio * (path[i] - path[i - 1])
                direction_vector = path[i] - path[i - 1]
                direction_vector = direction_vector.astype(float)  # Convert to float for division
                norm = np.linalg.norm(direction_vector)
                if norm != 0:
                    direction_vector /= norm  # Normalize if not zero
                return position, direction_vector
            accumulated_distance += segment_length

        # Fallback if we've finished the path
        direction_vector = path[-1] - path[-2] if len(path) > 1 else np.array([1, 0], dtype=float)
        direction_vector = direction_vector.astype(float)  # Convert to float for division
        norm = np.linalg.norm(direction_vector)
        if norm != 0:
            direction_vector /= norm  # Normalize if not zero
        return path[-1], direction_vector
    else:
        # Raise an error if the path array does not have the expected shape
        raise ValueError(f"Incorrect path shape: expected (n, 2), got {path.shape}")


def calculate_path(optimized_way_points, toff_aport, land_aport):
    """
        Calculate an optimized path for an aircraft from takeoff to landing, passing through several waypoints.
        The path is determined by choosing waypoints that do not deviate more than a specified angle from the
        current direction of travel, while also considering the distance to minimize the total travel cost.

        Args:
            optimized_way_points (np.ndarray): Array of waypoints to consider for the path.
            toff_aport (np.ndarray): Takeoff location coordinates.
            land_aport (np.ndarray): Landing location coordinates.

        Returns:
            np.ndarray: An array representing the calculated path from takeoff to landing.
        """
    path = [toff_aport]  # Start with the takeoff location
    while len(optimized_way_points) > 0:
        if len(path) > 1:
            current_direction = path[-1] - path[-2]
        else:
            current_direction = land_aport - toff_aport

        min_cost = float('inf')
        next_waypoint = None
        next_index = -1

        for index, waypoint in enumerate(optimized_way_points):
            travel_vector = waypoint - path[-1]
            travel_distance = np.linalg.norm(travel_vector)
            total_cost = travel_distance  # Simple cost function for now

            cosine_angle = np.dot(current_direction, travel_vector) / (np.linalg.norm(current_direction) * np.linalg.norm(travel_vector) + np.finfo(float).eps)
            angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

            if np.abs(angle) <= np.radians(40) and total_cost < min_cost:
                min_cost = total_cost
                next_waypoint = waypoint
                next_index = index

        if next_waypoint is not None:
            path.append(next_waypoint)
            optimized_way_points = np.delete(optimized_way_points, next_index, axis=0)
        else:
            break

    path.append(land_aport)  # Ensure landing airport is always added
    return np.array(path)


grid_size = 100
cov_scale = 25
Vx, Vy = generate_wind_data(100)
x = np.linspace(0, grid_size, grid_size)
y = np.linspace(0, grid_size, grid_size)
X, Y = np.meshgrid(x, y)
coordinates = np.vstack((X.ravel(), Y.ravel())).T

# Define precipitation sources with names, initial positions, and directions
precipitation_sources = {
    "Alpha": {"position": np.array([grid_size / 2, grid_size / 2]), "direction": np.array([1, 0])},
    "Beta": {"position": np.array([grid_size / 6, grid_size / 1.5]), "direction": np.array([-1, 0])}
}

way_points = np.random.rand(80, 2) * grid_size
airport_locations = np.random.rand(30, 2) * grid_size
toff_aport = airport_locations[2]
land_aport = airport_locations[9]


def forecast_source_positions_at_waypoints(waypoints, travel_times, aircraft_direction, precipitation_sources,
                                           proximity_threshold=10):
    """
    Forecasts precipitation sources' positions at the expected arrival times of waypoints and determines their relative directions.

    Args:
    waypoints (np.array): Array of waypoints the aircraft will traverse.
    travel_times (list): List of datetime objects representing the estimated times of arrival at each waypoint.
    aircraft_direction (np.array): The current travel direction of the aircraft.
    precipitation_sources (dict): Current positions and directions of precipitation sources.
    proximity_threshold (float): Distance threshold for considering a source in the vicinity.

    Returns:
    dict: Information about forecasted source positions, directions, and proximity at each waypoint.
    """
    source_forecasts = {}
    current_time = datetime.now()

    for waypoint, arrival_time in zip(waypoints, travel_times):
        forecasted_sources_info = {}
        time_delta = (arrival_time - current_time).total_seconds() / 3600.0  # time difference in hours

        for name, source in precipitation_sources.items():
            # Forecast the future position of the source
            forecasted_position = source['position'] + source['direction'] * time_delta

            # Vector from the waypoint to the forecasted source position
            vector_to_source = forecasted_position - waypoint
            distance = np.linalg.norm(vector_to_source)

            # Normalize for direction calculation
            vector_to_source_normalized = vector_to_source / distance if distance != 0 else vector_to_source

            # Dot product for relative direction
            direction_dot_product = np.dot(aircraft_direction, vector_to_source_normalized)

            # Cross product for side (right/left)
            cross_product_z = np.cross(aircraft_direction, vector_to_source_normalized)
            side = "left" if cross_product_z > 0 else "right"

            # Determine relative direction
            relative_direction = "towards" if direction_dot_product > 0 else "away"

            # Check proximity
            in_vicinity = distance <= proximity_threshold

            # Store the information
            forecasted_sources_info[name] = {
                "forecasted_position": forecasted_position,
                "distance": distance,
                "relative_direction": relative_direction,
                "side": side,
                "in_vicinity": in_vicinity
            }

        source_forecasts[waypoint.tobytes()] = forecasted_sources_info  # Use waypoint byte representation as key

    return source_forecasts


def calculate_optimized_waypoints(start_aport, land_aport, way_points, proxy_margin=2.0):
    """
    Calculate optimized waypoints from a starting point to a fixed landing airport.

    Args:
    start_aport (np.array): Starting waypoint, typically the takeoff airport.
    land_aport (np.array): Fixed landing airport.
    way_points (np.array): Array of potential waypoints.
    proxy_margin (float): Maximum perpendicular distance from the direct line to consider a waypoint.

    Returns:
    np.array: Array of optimized waypoints.
    """
    # Vector from start to landing airport
    takeoff_to_landing = land_aport - start_aport
    takeoff_to_landing_norm = np.linalg.norm(takeoff_to_landing)

    # Calculate perpendicular distances to the line from start to landing
    per_distances = np.abs(np.cross(takeoff_to_landing, start_aport - way_points)) / takeoff_to_landing_norm

    # Filter waypoints that are close to the line
    close_way_points = way_points[per_distances <= proxy_margin]

    # Further filter to ensure waypoints are in the direction of the landing airport
    vectors_to_waypoints = close_way_points - start_aport
    norms_to_waypoints = np.linalg.norm(vectors_to_waypoints, axis=1)
    boolean_mask = norms_to_waypoints <= (takeoff_to_landing_norm + proxy_margin)

    # Final list of optimized waypoints
    optimized_way_points = close_way_points[boolean_mask]

    print(f"Number of optimized waypoints: {len(optimized_way_points)}")
    return optimized_way_points


def adjust_waypoint_if_precip_high(path, travel_times, forecasts, precipitation_threshold, precipitation_sources):
    """
        Adjusts the aircraft's flight path based on real-time precipitation forecasts. Waypoints are shifted
        to avoid regions with precipitation above a given threshold.

        Args:
            path (list): List of tuples or np.ndarray containing the current flight path waypoints.
            travel_times (list): Estimated travel times to each waypoint.
            forecasts (dict): Dictionary with precipitation forecasts for each waypoint.
            precipitation_threshold (float): Threshold above which precipitation is considered high.
            precipitation_sources (dict): Information about precipitation sources used for forecasting.

        Returns:
            list: The adjusted path with waypoints modified to avoid high precipitation areas.
        """
    print("Original path:", path)
    new_path = []
    i = 0
    while i < len(path):
        pos = path[i]
        print(f"Processing waypoint {i}: {pos}")  # Debug line
        if isinstance(pos, (list, tuple, np.ndarray)) and len(pos) == 2:

            time = travel_times[i]
            forecast = forecasts.get((int(pos[0]), int(pos[1])), (None, 0))[1]

            if forecast > precipitation_threshold:
                print(f"High precipitation at waypoint {i}, adjusting...")
                # Calculate source direction and find adjustment direction
                nearest_source, source_direction = find_nearest_source(pos, precipitation_sources)
                # Create a circle around the current waypoint
                circle = Circle(pos, 2, fill=False, color='red', linestyle='--')
                ax.add_patch(circle)

                if i == 0:
                    # Use the direction from takeoff to landing airport for the initial direction
                    aircraft_direction = land_aport - toff_aport
                    aircraft_direction /= np.linalg.norm(aircraft_direction)  # Normalize the direction vector
                else:
                    aircraft_direction = path[i] - path[i - 1]
                    aircraft_direction /= np.linalg.norm(aircraft_direction)  # Normalize the direction vector

                # Calculate tangent points
                tangent_points = calculate_tangent_points(pos, 2, source_direction, aircraft_direction)
                # Decide on which side to focus based on source direction
                if np.cross(source_direction, np.array([1, 0])) > 0:  # Assuming upward direction as positive
                    chosen_point = tangent_points[1]  # Right side tangent point
                else:
                    chosen_point = tangent_points[0]  # Left side tangent point

                # Find new waypoint at 30 degrees from tangent
                tangent_vector = chosen_point - pos
                selected_waypoint = find_waypoint_within_angle(pos, way_points, tangent_vector, 40)
                # new_path.append(selected_waypoint)
                # Re-calculate travel times and forecasts for the new path segment
                partial_path = np.concatenate(([selected_waypoint], path[i + 1:]))

                partial_travel_times = calculate_travel_times(partial_path, speed_kmph=300)
                partial_forecasts = forecast_precipitation_along_path(partial_path, partial_travel_times, coordinates,
                                                                      grid_size, cov_scale, Vx, Vy,
                                                                      precipitation_sources)
                print('selected waypoint', selected_waypoint)
                print('Type of selected waypoint:', type(selected_waypoint))
                print('slice path', path[i + 1:])
                print('Type of slice path:', type(path[i + 1:]))
                print('partial path', partial_path)
                print('Type of partial path', type(partial_path))
                new_path = new_path[:i]  # Retain the portion of new_path up to the current index
                # Extend new_path with the new partial path
                new_path.extend(partial_path)  # Adding all waypoints from the adjusted one onwards

                i += len(partial_path) - 1  # Adjust index to skip recalculated part
            else:
                new_path.append(pos)

        else:
            print(f"Invalid waypoint encountered at index {i}: {pos}")
            # Handle invalid data, perhaps skip or use a fallback position
        i += 1

    print("New path after adjustments:", new_path)
    return new_path


def find_waypoint_within_angle(current_position, waypoints, tangent_vector, angle_deg=40):
    """
    Find waypoints within a specified angle from the tangent vector.

    Args:
    current_position (np.array): The current position.
    waypoints (list of np.array): List of available waypoints.
    tangent_vector (np.array): The tangent direction vector.
    angle_deg (float): The angle constraint in degrees.

    Returns:
    np.array: The selected waypoint within the angle constraint, or None if no such waypoint exists.
    """
    # Normalize the tangent vector
    tangent_vector = tangent_vector / np.linalg.norm(tangent_vector)
    angle_rad = np.radians(angle_deg)
    selected_waypoint = None
    min_distance = float('inf')

    for waypoint in waypoints:
        # Calculate the vector from the current position to the waypoint
        direction_to_waypoint = waypoint - current_position
        norm = np.linalg.norm(direction_to_waypoint)
        if norm == 0:
            continue  # Skip this waypoint if it's the same as the current position
        direction_to_waypoint_normalized = direction_to_waypoint / norm

        # Calculate the angle between the tangent vector and the waypoint direction vector
        dot_product = np.dot(tangent_vector, direction_to_waypoint_normalized)
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))

        # Check if the waypoint is within the specified angle
        if angle <= angle_rad:
            distance = np.linalg.norm(direction_to_waypoint)
            if distance < min_distance:
                min_distance = distance
                selected_waypoint = waypoint
                print('selected waypoint', selected_waypoint)
                print('ran')

    return selected_waypoint


def calculate_tangent_points(center, radius, incoming_direction, aircraft_direction):
    """
    Calculate the points of tangency to a circle given a direction vector pointing towards the circle.
    It also considers the aircraft's direction to choose the appropriate tangent point based on the relative direction of the source.

    Args:
    center (np.array): The center of the circle (position of the source).
    radius (float): The radius of the circle around the source.
    incoming_direction (np.array): The movement direction of the source.
    aircraft_direction (np.array): The current movement direction of the aircraft.

    Returns:
    np.array: The selected tangent point based on the relative movement of the source.
    """
    # Normalize the direction vectors
    norm_incoming_direction = incoming_direction / np.linalg.norm(incoming_direction)
    norm_aircraft_direction = aircraft_direction / np.linalg.norm(aircraft_direction)

    # Calculate the angle for the tangent points (90 degrees in radians)
    angle = np.pi / 2

    # Define rotation matrices for both directions
    rotate_clockwise = np.array([[np.cos(angle), np.sin(angle)],
                                 [-np.sin(angle), np.cos(angle)]])
    rotate_counterclockwise = np.array([[np.cos(-angle), np.sin(-angle)],
                                        [-np.sin(-angle), np.cos(-angle)]])

    # Calculate tangent vectors
    tangent_vector1 = np.dot(rotate_clockwise, norm_incoming_direction) * radius
    tangent_vector2 = np.dot(rotate_counterclockwise, norm_incoming_direction) * radius

    # Calculate tangent points
    point1 = center + tangent_vector1
    point2 = center + tangent_vector2

    # Determine the direction of the source relative to the aircraft's direction
    cross_product = np.cross(norm_aircraft_direction, norm_incoming_direction)

    # Select the tangent point based on the direction of the source movement relative to the aircraft
    if cross_product > 0:
        return point2  # Use right tangent if source is to the right of the aircraft's direction
    else:
        return point1  # Use left tangent if source is to the left


# Initialize the figure and axis
fig, ax = plt.subplots()
img = ax.imshow(np.zeros((grid_size, grid_size)), cmap='Blues', origin='lower', extent=[0, grid_size, 0, grid_size])
plt.colorbar(img, ax=ax, label='Precipitation Intensity')

# Triangle representing the aircraft (initially placed at the starting point)
triangle = Polygon([[-1, -1]], closed=True, color='orange')  # Initial dummy points
ax.add_patch(triangle)

optimized_way_points = calculate_optimized_waypoints(toff_aport, land_aport, way_points)
# Calculate path before the animation
path = calculate_path(optimized_way_points, toff_aport, land_aport)

# Plot static elements (waypoints, airports, and path) before starting the animation
ax.scatter(way_points[:, 0], way_points[:, 1], color='blue', s=50, label='Waypoints', zorder=5)
ax.scatter(airport_locations[:, 0], airport_locations[:, 1], color='red', s=100, label='Airports', zorder=5)
ax.plot(path[:, 0], path[:, 1], 'g-', label='Path', zorder=4)  # Ensure path is below waypoints and airports


# This function is called within the animation loop to update the triangle
def update_aircraft(triangle, position, direction_vector):
    """
        Updates the position and orientation of an aircraft represented as a triangle on a plot.

        Args:
            triangle (matplotlib.patches.Polygon): The triangle object representing the aircraft in the plot.
            position (np.array): The current position of the aircraft's center in 2D space.
            direction_vector (np.array): The vector representing the aircraft's forward direction.

        Details:
            The aircraft is visually represented by a triangle whose orientation corresponds to its direction of travel.
            The triangle's 'nose' points in the direction of the vector, and its 'base' is perpendicular to this direction.
        """
    aircraft_length = 3.0  # Length of the aircraft triangle
    aircraft_width = 1.5   # Width of the aircraft triangle

    # Calculate the normalized direction vector
    dir_vec_norm = direction_vector / np.linalg.norm(direction_vector)

    # Calculate the perpendicular vector (for the base of the triangle)
    perp_vec = np.array([-dir_vec_norm[1], dir_vec_norm[0]])

    # Calculate the vertices of the triangle
    nose = position + dir_vec_norm * (aircraft_length / 2)
    left_vertex = position - dir_vec_norm * (aircraft_length / 2) + perp_vec * (aircraft_width / 2)
    right_vertex = position - dir_vec_norm * (aircraft_length / 2) - perp_vec * (aircraft_width / 2)

    print(f"Updating aircraft to position {position} with direction {direction_vector}")

    # Update the triangle's vertices
    triangle.set_xy([left_vertex, right_vertex, nose])


def find_nearest_source(aircraft_position, precipitation_sources):
    """
        Finds the nearest precipitation source to a given aircraft position.

        Args:
            aircraft_position (np.array): The current 2D position of the aircraft.
            precipitation_sources (dict): A dictionary of precipitation sources, where each key is a name of the source
                                          and each value is a dict containing 'position' and 'direction' of the source.

        Returns:
            tuple: The name of the nearest source and the direction vector of this source.

        Description:
            This function iterates through all available precipitation sources, calculates the Euclidean distance from
            the aircraft to each source, and returns the nearest source and its direction vector. This is useful for
            avoidance maneuvers or navigation adjustments in response to adverse weather conditions.
        """
    min_distance = float('inf')
    nearest_source = None
    nearest_direction = None

    for name, source in precipitation_sources.items():
        distance = np.linalg.norm(source["position"] - aircraft_position)
        if distance < min_distance:
            min_distance = distance
            nearest_source = name
            nearest_direction = source["direction"]

    return nearest_source, nearest_direction


def calculate_wind_vector(frame, Vx_mean, Vy_mean, modulation_strength=0.1, frequency=40.0):

    """Calculate wind vector for a given frame using trigonometric modulation."""
    time_factor = modulation_strength * np.array([np.sin(frame / frequency), np.cos(frame / frequency)])
    return np.array([Vx_mean, Vy_mean]) * time_factor


def calculate_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return np.linalg.norm(point1 - point2)


def update_plot(frame, img, coordinates, path, airport_locations, grid_size, cov_scale, Vx, Vy, precipitation_sources, text_annotations):
    """
        Updates the plot for a simulation or visualization, adjusting for dynamic environmental conditions and aircraft movement.

        Args:
            frame (int): Current frame number in the simulation.
            img (matplotlib.image.AxesImage): Image object showing precipitation data.
            coordinates (np.array): Array of coordinates used in the simulation.
            path (list): Current path of the aircraft as a list of coordinates.
            airport_locations (dict): Dictionary with locations of airports.
            grid_size (int): Size of the grid used in the simulation for precipitation and other calculations.
            cov_scale (float): Scale factor for covariance in precipitation calculation.
            Vx, Vy (np.array): Arrays containing the x and y components of wind vectors.
            precipitation_sources (dict): Dictionary of sources generating precipitation.
            text_annotations (list): List of matplotlib.text.Text objects used for displaying data on the plot.

        Returns:
            tuple: Tuple containing updated image and aircraft triangle, representing new state of the plot.
        """
    # Assuming 200 frames total, calculate time_factor and wind_vector for the precipitation movement
    Vx_mean, Vy_mean = Vx.mean(), Vy.mean()
    wind_vector = calculate_wind_vector(frame, Vx_mean, Vy_mean)

    # Update source directions based on wind_vector
    for source in precipitation_sources.values():
        source["direction"] = wind_vector

    # Calculate new centers for the precipitation field
    centers = [np.array([grid_size / 2, grid_size / 2]) + wind_vector * frame,
               np.array([grid_size / 6, grid_size / 1.5]) - wind_vector * frame]
    centers = [np.clip(center, 0, grid_size - 1) for center in centers]

    timestamp = datetime.now() + timedelta(minutes=frame)  # timestamp calculation / I've used here min for 1 frame

    # Update the precipitation field
    precipitation_field = simulate_precipitation(coordinates, cov_scale, timestamp, precipitation_sources)

    # Set data for the updated precipitation field
    img.set_data(precipitation_field)
    img.set_clim(np.min(precipitation_field), np.max(precipitation_field))

    print("Final path array before processing:", path)  # Debug statement

    # Get the current position and direction vector for the aircraft
    # Get the current position and direction vector for the aircraft
    try:
        aircraft_position, direction_vector = get_aircraft_position_and_angle(path, frame, 200)
    except ValueError as e:
        print(f"Error in path data at frame {frame}: {e}")
    nearest_source, nearest_direction = find_nearest_source(aircraft_position, precipitation_sources)

    # Record the timestamp and position of the aircraft
    record_aircraft_timestamps(aircraft_position, timestamp)

    optimized_way_points = calculate_optimized_waypoints(toff_aport, land_aport, way_points)

    # Calculate path
    path = calculate_path(optimized_way_points, toff_aport, land_aport)
    travel_times = calculate_travel_times(path, speed_kmph=300)  # Speed in km/h

    # Get precipitation forecasts along the path
    precipitation_forecasts = forecast_precipitation_along_path(path, travel_times, coordinates, grid_size, cov_scale,

                                                                Vx, Vy, precipitation_sources)

    # Calculate and update distances for each source
    distance_texts = []
    for name, source in precipitation_sources.items():
        distance = calculate_distance(aircraft_position, source["position"])
        distance_texts.append(f"{name}: {distance:.2f} units")

    # Update text annotations on the plot
    for text, annotation in zip(distance_texts, text_annotations):
        annotation.set_text(text)

    # Update the aircraft's position and orientation using the direction vector
    update_aircraft(triangle, aircraft_position, direction_vector)

    # Return the updated artists
    return img, triangle


# Initialize text annotations
text_annotations = [ax.text(0.02, 0.98 - i*0.04, '', transform=ax.transAxes, fontsize=10, verticalalignment='top') for i in range(len(precipitation_sources))]

travel_times = calculate_travel_times(path, speed_kmph=300)

# Forecast and adjust path once before the animation
forecasts = forecast_precipitation_along_path(path, travel_times, coordinates, grid_size, cov_scale, Vx, Vy, precipitation_sources)
adjusted_path = adjust_waypoint_if_precip_high(path, travel_times, forecasts, precipitation_threshold, precipitation_sources)

# Convert adjusted_path to a numpy array
adjusted_path_np = np.array(adjusted_path)
# Plot the new path in red
ax.plot(adjusted_path_np[:, 0], adjusted_path_np[:, 1], 'r-', label='Adjusted path')
# Now start the animation using the adjusted path
ani = FuncAnimation(fig, update_plot, frames=200, fargs=(img, coordinates, adjusted_path, airport_locations, grid_size, cov_scale, Vx, Vy, precipitation_sources, text_annotations), repeat=True)
plt.legend()
plt.show()