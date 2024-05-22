#####################################################################################
#                                                                                    #
#                       Flight Planning Code                                         #
#                          using Folium                                              #
#                                                                                    #
# Author: [Luca Mattiocco]                                                           #
# Date: April 2024                                                                   #
# Affiliation: Cranfield University                                                  #
#                                                                                    #
# Description:                                                                       #
# This code utilizes Folium to visualize flight planning, allowing users to choose   #
# departure and arrival airports, select runways, and generate flight paths based    #
# on the chosen criteria. It calculates distances between airports and checks if     #
# the flight is feasible. Additionally, it saves flight plans and airport            #
# information to CSV files.                                                          #
#                                                                                    #
#                                                                                    #
#                                                                                    #
#####################################################################################


import folium
import pandas as pd
from geopy.distance import geodesic

# Load data from the CSV file with semicolon delimiter
data = pd.read_csv("Airports_Pool.csv", delimiter=';')

# Remove rows with NaN values
data = data.dropna()

# Create a Google Map centered on the UK
center_lat = 54.702354
center_lon = -3.276575
mymap = folium.Map(location=[center_lat, center_lon], zoom_start=6)  # Zoom level adjusted for better view of the UK

# Define colors for departure and arrival markers
departure_color = 'green'
arrival_color = 'red'

# Function to calculate midpoint coordinates
def calculate_midpoint(start_lat, start_lon, end_lat, end_lon):
    midpoint_lat = (start_lat + end_lat) / 2
    midpoint_lon = (start_lon + end_lon) / 2
    return midpoint_lat, midpoint_lon

# Function to calculate distance between two points
def calculate_distance(start_lat, start_lon, end_lat, end_lon):
    start_point = (start_lat, start_lon)
    end_point = (end_lat, end_lon)
    return geodesic(start_point, end_point).kilometers

# Initialize lists to store takeoff and landing information
takeoff_info = []
landing_info = []

while True:
    departure_airport = None
    arrival_airport = None
    
    # Ask user to choose departure airport
    print("\nChoose departure airport:")
    for i, airport in enumerate(data['map'].unique(), 1):
        print(f"{i}. {airport}")
    departure_choice = int(input("Enter the number corresponding to the departure airport: "))
    departure_airport = data['map'].unique()[departure_choice - 1]
    takeoff_info.append((f"Takeoff - {departure_airport}", data.loc[data['map'] == departure_airport, 'Start Latitude'].iloc[0], data.loc[data['map'] == departure_airport, 'Start Longitude'].iloc[0]))

    # Check if the departure airport has multiple runways
    departure_data = data[data['map'] == departure_airport]
    if len(departure_data) == 1:
        departure_runway = departure_data['Runway Name'].iloc[0]
    else:
        print(f"Multiple runways found for {departure_airport}. Please choose a runway.")
        for i, runway in enumerate(departure_data['Runway Name'], 1):
            print(f"{i}. {runway}")
        runway_choice = int(input("Enter the number corresponding to the chosen runway: "))
        departure_runway = departure_data['Runway Name'].iloc[runway_choice - 1]

    # Extract heading direction from the runway name and convert it to degrees
    departure_runway_parts = departure_runway.split('/')
    departure_heading_choices = [(int(part.lstrip('0').rstrip('L').rstrip('R')) * 10) for part in departure_runway_parts]
    print(f"Choose the heading direction for departure from runway {departure_runway}: {departure_heading_choices}")
    departure_heading_choice = int(input())
    departure_heading = departure_heading_choice

    # Ask user to choose arrival airport
    print("\nChoose arrival airport:")
    for i, airport in enumerate(data['map'].unique(), 1):
        print(f"{i}. {airport}")
    arrival_choice = int(input("Enter the number corresponding to the arrival airport: "))
    arrival_airport = data['map'].unique()[arrival_choice - 1]
    landing_info.append((f"Landing - {arrival_airport}", data.loc[data['map'] == arrival_airport, 'Start Latitude'].iloc[0], data.loc[data['map'] == arrival_airport, 'Start Longitude'].iloc[0]))

    # Check if the arrival airport has multiple runways
    arrival_data = data[data['map'] == arrival_airport]
    if len(arrival_data) == 1:
        arrival_runway = arrival_data['Runway Name'].iloc[0]
    else:
        print(f"Multiple runways found for {arrival_airport}. Please choose a runway.")
        for i, runway in enumerate(arrival_data['Runway Name'], 1):
            print(f"{i}. {runway}")
        runway_choice = int(input("Enter the number corresponding to the chosen runway: "))
        arrival_runway = arrival_data['Runway Name'].iloc[runway_choice - 1]

    # Extract heading direction from the runway name and convert it to degrees
    arrival_runway_parts = arrival_runway.split('/')
    arrival_heading_choices = [(int(part.lstrip('0').rstrip('L').rstrip('R')) * 10) for part in arrival_runway_parts]
    print(f"Choose the heading direction for arrival to runway {arrival_runway}: {arrival_heading_choices}")
    arrival_heading_choice = int(input())
    arrival_heading = arrival_heading_choice

    # Calculate distance between departure and arrival airports
    distance = calculate_distance(departure_data.iloc[0]['Start Latitude'], departure_data.iloc[0]['Start Longitude'],
                                  arrival_data.iloc[0]['Start Latitude'], arrival_data.iloc[0]['Start Longitude'])

    # If distance is less than 200 km, prompt user to choose another arrival airport
    if distance < 200:
        print(f"The distance between {departure_airport} and {arrival_airport} is less than 200 km, so the flight is not possible.")
        continue
    else:
        # Calculate midpoint coordinates
        mid_lat, mid_lon = calculate_midpoint(departure_data.iloc[0]['Start Latitude'], departure_data.iloc[0]['Start Longitude'], arrival_data.iloc[0]['Start Latitude'], arrival_data.iloc[0]['Start Longitude'])

        # Display flight plan for confirmation
        print("\nFlight Plan:")
        print(f"From {departure_airport}, take off from Runway {departure_runway} heading {departure_heading}° to {arrival_airport}, coming from heading {arrival_heading}° to land on Runway {arrival_runway}.")

        confirm = input("Is this correct? (yes/no): ").lower()

        # If confirmed, add markers for departure and arrival points and exit loop
        if confirm == 'yes':
            # Add departure marker
            folium.Marker(location=[departure_data.iloc[0]['Start Latitude'], departure_data.iloc[0]['Start Longitude']], popup=f"Takeoff - {departure_airport} Runway {departure_runway} Heading to {arrival_airport}", icon=folium.Icon(color=departure_color)).add_to(mymap)
            
            # Add arrival marker
            folium.Marker(location=[arrival_data.iloc[0]['Start Latitude'], arrival_data.iloc[0]['Start Longitude']], popup=f"Landing - {arrival_airport} Runway {arrival_runway} From {departure_airport}", icon=folium.Icon(color=arrival_color)).add_to(mymap)
            
            print("Flight plan confirmed. Thank you for using the program.")
            break
        elif confirm == 'no':
            print("Restarting flight planning process...")
            continue
        else:
            print("Invalid choice. Please enter either 'yes' or 'no'.")

    continue

# Save the map to an HTML file
mymap.save("Airport_Runways_Selected.html")

# Write takeoff and landing information to a CSV file
columns = ['Point', 'Latitude ', 'Longitude']

takeoff_name = [info[0] for info in takeoff_info]
takeoff_lat = [info[1] for info in takeoff_info]
takeoff_lon = [info[2] for info in takeoff_info]

landing_name = [info[0] for info in landing_info]
landing_lat = [info[1] for info in landing_info]
landing_lon = [info[2] for info in landing_info]

takeoff_df = pd.DataFrame({'Point': takeoff_name, 'Latitude': takeoff_lat, 'Longitude': takeoff_lon})
landing_df = pd.DataFrame({'Point': landing_name, 'Latitude': landing_lat, 'Longitude': landing_lon})

# Write takeoff and landing information to a CSV file
takeoff_df.to_csv('Takeoff_info.csv', index=False, sep=',')
landing_df.to_csv('Landing_info.csv', index=False, sep=',')

