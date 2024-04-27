#%%
from class_definition import OptimisePath
import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np
import folium
from folium.plugins import MarkerCluster
from shapely.geometry import MultiPoint, Point, Polygon
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from geographiclib.geodesic import Geodesic
from geopy.distance import distance, geodesic
import geopandas as gpd
import random
import heapq

waypoints=pd.read_csv("waypoints.csv",delimiter=';')
waypoints = waypoints.dropna()
coordinates = waypoints[["Latitude", "Longitude"]].values
for i in range(len(coordinates)):

    coordinates[i,0]=float(coordinates[i,0])
    coordinates[i,1]=float(coordinates[i,1])
coordinates_rad=coordinates*np.pi/180
db = DBSCAN(eps=0.002, min_samples=1, algorithm='ball_tree', metric='haversine').fit(coordinates_rad)
pts=db.labels_
num_clusters = len(set(pts)) - (1 if -1 in pts else 0)

clusters_indices=[]
cluster_coordinates=[]
total=0
for i in range(num_clusters):
    cluster=np.transpose(np.where(pts==i)[0])
    cluster_coordinates.append(waypoints.iloc[cluster])
    total=total+len(cluster)


final_waypoints=pd.DataFrame()
tolerance=0.001
for cluster in cluster_coordinates:

    cluster['Latitude'] = cluster['Latitude'].astype(float)
    cluster['Longitude'] = cluster['Longitude'].astype(float)
    if len(cluster)>1:
        points=MultiPoint(np.array(cluster[["Latitude", "Longitude"]].values))
        rep=points.representative_point()
        df = cluster[
            (abs(cluster["Latitude"] - rep.x) < tolerance) &
            (abs(cluster["Longitude"] - rep.y) < tolerance)
        ]
    else:
        df=cluster
    
    final_waypoints= pd.concat([final_waypoints, df], axis=0)
m=folium.Map()
for i in range(len(final_waypoints)):
    lat= final_waypoints.iloc[i]["Latitude"]
    lon= final_waypoints.iloc[i]["Longitude"]
    folium.CircleMarker(location=[lat,lon],radius=5,color="black",fill=True,fill_color="black",popup=final_waypoints.iloc[i]["Point"]).add_to(m)



landing_point=(55.863676,-4.448690)
takeoff_point=(51.464786,-0.486215)
folium.CircleMarker(location=takeoff_point,radius=5,color="blue",fill=True,fill_color="blue",popup="Take off point").add_to(m)
folium.CircleMarker(location=landing_point,radius=5,color="blue",fill=True,fill_color="blue",popup="Landing point").add_to(m)
m.save("lol.html")

original_route=["take_off","SILVA","CLIPY","MAPLE","STAFA","NANTI","ASNIP","TADAL","SUBUK","DCS19","DCS36","SUMIN","FENIK","landing"]

max_distance=50
def CalculateBearing(heading,angle):
    new_bearing=heading+angle #angle needs to be negative if going anticlockwise!!!
    if new_bearing <0 :
        new_bearing=new_bearing+360
    return new_bearing
bounds=[]
left_bounds=[]
right_bounds=[]
for i in range(len(original_route)-1):
    
    if original_route[i]=="take_off":
        lat1=takeoff_point[0]
        lon1=takeoff_point[1]
    else:
        df=final_waypoints[final_waypoints["Point"]==original_route[i]].reset_index(drop=True)
        lat1=df.iloc[0]["Latitude"]
        lon1=df.iloc[0]["Longitude"]
    if i+1==len(original_route)-1:
        lat2=landing_point[0]
        lon2=landing_point[1]

    else:
        df=final_waypoints[final_waypoints["Point"]==original_route[i+1]].reset_index(drop=True)
        lat2=df.iloc[0]["Latitude"]
        lon2=df.iloc[0]["Longitude"]

    geod = Geodesic.WGS84.Inverse(lat1, lon1, lat2, lon2)
    bearing = geod['azi1']

    left_bound= distance(kilometers=max_distance).destination((lat1,lon1),CalculateBearing(bearing,-90))
    left_bounds.append((left_bound.longitude,left_bound.latitude))
    right_bound=distance(kilometers=max_distance).destination((lat1,lon1),CalculateBearing(bearing,+90))
    right_bounds.append((right_bound.longitude,right_bound.latitude))

left_bound=distance(kilometers=max_distance).destination(landing_point,CalculateBearing(bearing,-90))
left_bounds.append((left_bound.longitude,left_bound.latitude))
right_bound=distance(kilometers=max_distance).destination(landing_point,CalculateBearing(bearing,+90))
right_bounds.append((right_bound.longitude,right_bound.latitude))
right_bounds.reverse()
bounds=left_bounds+right_bounds
region_polygon = Polygon(bounds)
polygon_df = gpd.GeoDataFrame(geometry=[region_polygon])
polygon_df.plot()
plt.show()
final_waypoints_filtered=final_waypoints.reset_index(drop=True)

for i in range(len(final_waypoints)):
    lat=final_waypoints.iloc[i]["Latitude"]
    lon=final_waypoints.iloc[i]["Longitude"]
    point_to_check = Point(lon, lat)
    is_within_region = polygon_df.contains(point_to_check).values[0]
    if is_within_region==False:
        final_waypoints_filtered=final_waypoints_filtered.drop(i)
polygonmap=folium.Map()
final_waypoints_filtered=final_waypoints_filtered.reset_index(drop=True)
for i in range(len(bounds)):
    folium.Marker(location=(bounds[i][1], bounds[i][0]),icon=folium.Icon(color='red'),popup=f"{i+1}").add_to(polygonmap)
for i in range(len(final_waypoints_filtered)):
    lat=final_waypoints_filtered.iloc[i]["Latitude"]
    lon=final_waypoints_filtered.iloc[i]["Longitude"]
    folium.Marker(location=[lat,lon],icon=folium.Icon(color='blue'),popup=final_waypoints_filtered.iloc[i]["Point"]).add_to(polygonmap)

polygonmap
#%%
def weighted_distance(point1, point2, weight):
    dist=geodesic(point1,point2).kilometers+weight
    return dist
all_points=np.vstack((final_waypoints_filtered[["Latitude","Longitude"]].values,np.array(takeoff_point),np.array(landing_point)))
tri = Delaunay(all_points)
graph = {}
use_random_weights="yes"
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

graph.keys()
graph[(51.470556, -1.049167)]
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
shortest_path = dijkstra(graph, takeoff_point, landing_point)
optimised_path=pd.DataFrame()
for waypoint in shortest_path:
    df = final_waypoints_filtered[
            (abs(final_waypoints_filtered["Latitude"] - waypoint[0]) < tolerance) &
            (abs(final_waypoints_filtered["Longitude"] - waypoint[1]) < tolerance)

    ]
    optimised_path=pd.concat([optimised_path,df],axis=0)
#%%
# final_path=np.vstack((takeoff_point,optimised_path[["Latitude","Longitude"]].values,landing_point))
takeoff_point_dict={
    "Latitude": takeoff_point[0],
    "Longitude": takeoff_point[1],
    "Point": "Takeoff"
}
landing_point_dict={
    "Latitude": landing_point[0],
    "Longitude": landing_point[1],
    "Point": "Landing"
}
optimised_path = pd.concat([pd.DataFrame(takeoff_point_dict, index=[0]), optimised_path], ignore_index=True)

# Add row at the bottom
landing_point_df=pd.DataFrame(landing_point_dict,index=[len(shortest_path)-1])
optimised_path = pd.concat([optimised_path,landing_point_df], ignore_index=True)
# df = pd.concat([df, new_row_bottom], ignore_index=True)
# df=final_waypoints[final_waypoints["Point"]==original_route].reset_index(drop=True)
mask = final_waypoints_filtered['Point'].isin(original_route)
filtered_df = final_waypoints_filtered[mask]
filtered_df = filtered_df.set_index('Point').loc[original_route[1:-1]].reset_index()
# filtered_df = final_waypoints_filtered[final_waypoints_filtered['Point'].isin(original_route)]

filtered_df = pd.concat([pd.DataFrame(takeoff_point_dict, index=[0]), filtered_df], ignore_index=True)

# Add row at the bottom
# landing_point_df=pd.DataFrame(landing_point_dict,index=[len(shortest_path)-1])
filtered_df = pd.concat([filtered_df,landing_point_df], ignore_index=True)
original_route=filtered_df
#%%
map_optim_vs_original=folium.Map()
for i in range(len(original_route)):
    if i in range(1,len(original_route)-1):
        lat=original_route.iloc[i]["Latitude"]
        lon=original_route.iloc[i]["Longitude"]

        folium.CircleMarker(location=[lat,lon],radius=5,color="black",fill=True,fill_color="black",popup=original_route.iloc[i]["Point"]).add_to(map_optim_vs_original)
    
folium.PolyLine(locations=original_route[["Latitude","Longitude"]].values,color="black").add_to(map_optim_vs_original)

for i in range(len(optimised_path)):

    lat=optimised_path.iloc[i]["Latitude"]
    lon=optimised_path.iloc[i]["Longitude"]
    if i==0 or i==len(optimised_path)-1:
        folium.CircleMarker(location=[lat,lon],radius=5,color="blue",fill=True,fill_color="black",popup=optimised_path.iloc[i]["Point"]).add_to(map_optim_vs_original)
    else:
        folium.CircleMarker(location=[lat,lon],radius=5,color="red",fill=True,fill_color="black",popup=optimised_path.iloc[i]["Point"]).add_to(map_optim_vs_original)

folium.PolyLine(locations=optimised_path[["Latitude","Longitude"]].values,color="red").add_to(map_optim_vs_original)
map_optim_vs_original.save("optimisation example.html")

# %%