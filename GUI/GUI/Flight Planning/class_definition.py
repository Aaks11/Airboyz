
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
from pyproj import Geod
from folium.plugins import HeatMap
from shapely.geometry import LineString

'''
   +---------------------------------------------------+
   | Code Developed by                                 |
   | Luca Mattiocco and Jade Nassif                    |
   | Aerospace Computational Engineering MSc 2023-2024 |
   | Cranfield University                              |
   | luca.mattiocco@cranfield.ac.uk                    |
   | jade.nassif.385@cranfield.ac.uk                   |
   +---------------------------------------------------+ 
'''
''' 
This code contains the definion of the OptimisePath class
it is built according to the following four steps

1. Initialisation --> import the database of waypoints
2. Reduce the total number of waypoints by means of clustering
3. Define a preliminary/original route and filter the waypoints further by defining a corridor around it
4. Generate a Delaunay mesh with the final waypoints
5. Weight assignment
6. Running Djikstra's algorithm to obtain the optimised path based on weather (precipitation)
'''
class OptimisePath():
    '''
    Constructor
    Input is a pandas dataframe with the following columns:
    Point, Latitude, Longitude
    Where Point refers to the waypoint name
    '''
    def __init__(self,waypoints_df):

        #set default values for epsilon and corridor distance
        self.eps=0.002
        self.maximum_distance_away_from_route=50 #km

        #store df as class attribute
        self.waypoints_database=waypoints_df

    '''
    function which clusters the waypoints

    no input required. "eps" is an optional input defining the 
    "maximum distance between two samples for one to be considered as in the neighborhood of the other"
    in the DBSCAN algorith, by default 0.002
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html

    returns:
    - A dictionary containing information regarding the results of the clustering
    - An updated waypoints dataframe containing the filtered waypoints
    - A map displaying the remaining waypoints where the popup of each point is the waypoint name
    '''
    def ClusterWaypoints(self,**kwargs):

        # take in optional eps input if present in **kwargs
        if "eps" in kwargs.keys():
            self.eps = kwargs.get('eps')

        # extract coordinates into array
        coordinates = self.waypoints_database[["Latitude", "Longitude"]].values
        for i in range(len(coordinates)):

            #convert to float in case interpreted as strings
            coordinates[i,0]=float(coordinates[i,0])
            coordinates[i,1]=float(coordinates[i,1])

        #convert lats and lons into radians for haversine metric in DBSCAN
        coordinates_rad=coordinates*np.pi/180
        db = DBSCAN(eps=self.eps, min_samples=1, algorithm='ball_tree', metric='haversine').fit(coordinates_rad)

        #pts contains the cluster number of each waypoint, -1 if noise but no noise as min_samples=1
        pts=db.labels_

        # get number of clusters (starts at 0)
        num_clusters=int(np.max(pts))+1
        
        # initialise a list which wil contain the slice of the waypoints database corresponding to each cluster
        cluster_coordinates=[]

        for i in range(num_clusters):

            #extract indices of the points in the ith cluster 
            cluster=np.transpose(np.where(pts==i)[0])

            #add the slice of the waypoints database corresponding to the ith cluster to the list
            cluster_coordinates.append(self.waypoints_database.iloc[cluster])
        
        #initialise the updated dataframe
        clustered_waypoints=pd.DataFrame()
        #set a low tolerance to match coordinates to those in original waypoints database
        tolerance=0.000000001

        #loop through the list of clusters
        for cluster in cluster_coordinates:

            cluster['Latitude'] = cluster['Latitude'].astype(float)
            cluster['Longitude'] = cluster['Longitude'].astype(float)

            #if the cluster has more than one point, find the representative point
            if len(cluster)>1:

                #create a multipoint object from the coordinates of the cluster
                points=MultiPoint(np.array(cluster[["Latitude", "Longitude"]].values))

                #obtain the representative point
                rep=points.representative_point()

                #find the points in the cluster that are within the tolerance of the representative point
                #(should be only one except if any duplicates are present)
                df = cluster[
                    (abs(cluster["Latitude"] - rep.x) < tolerance) &
                    (abs(cluster["Longitude"] - rep.y) < tolerance)
                ]
                if len(df)>1:

                    #only retain the last row of the dataframe if there are duplicates
                    df.drop(df.index.max(), axis=0, inplace=True)

            else:
                
                #case where there is only one point in the cluster
                df=cluster
            
            #append the dataframe
            clustered_waypoints= pd.concat([clustered_waypoints, df], axis=0)
        
        # initialise the folium map object
        m=folium.Map(location=[55, -0.633950],zoom_start=5)

        #plot the clustered points
        for i in range(len(clustered_waypoints)):
            lat= clustered_waypoints.iloc[i]["Latitude"]
            lon= clustered_waypoints.iloc[i]["Longitude"]
            popup = f"""
                <div style="font-size: 15px;">
                    {clustered_waypoints.iloc[i]["Point"]}
                </div>
            """
            folium.CircleMarker(location=[lat,lon],radius=1.5,color="black",fill=True,fill_color="black",popup=popup).add_to(m)

        # store the dataframe    
        self.clustered_waypoints=clustered_waypoints

        #store the clustering into, good if no of clusters= no of rep points
        self.clustering_info={
            "No of clusters": num_clusters,
            "No of representative points": len(clustered_waypoints),
            "eps": self.eps
        }
        
        return self.clustering_info, self.clustered_waypoints, m

    '''
    Function which filters the waypoints further after defining a corridor around the original route
    Inputs:
    - original_route is a list of strings formatted as such ["take_off",...,"landing"] where 
    each string between the first and last entries is a waypoint name
    e.g: ["take_off","SILVA","CLIPY","MAPLE","STAFA","NANTI","ASNIP","TADAL","SUBUK","DCS19","SUMIN","FENIK","landing"]
    - takeoff_point and landing_point are tuples of two floats (lat,lon)
    - as an optional input one can specify the distance away from the original route defining the travel corridor, 50 km by default 

    Returns 
    - dataframe of the original route
    - a dataframe containing the final waypoints database
    - a map which shows the travel corridor and waypoints within it
    '''
    def FilterWaypoints(self,original_route,takeoff_point,landing_point,**kwargs):

        #check for any optional inputs
        if "maximum_distance_away_from_route" in kwargs.keys():
            self.maximum_distance_away_from_route = kwargs.get('maximum_distance_away_from_route')
        
        # initialise the original route as a dictionary (for conversion to pd df later on)
        original_route_dict={
            "Latitude": [],
            "Longitude": [],
            "Point": []
        }

        #extract the required variables from the class attributes
        clustered_waypoints=self.clustered_waypoints
        max_distance=self.maximum_distance_away_from_route
    
        #initialise lists which will contain the bounds of the polygon defining the corridor
        bounds=[]
        left_bounds=[]
        right_bounds=[]

        #loop through the original route, except for landing point
        for i in range(len(original_route)-1):
            
            # populate the original route dictionary
            if original_route[i]=="take_off":
                lat1=takeoff_point[0]
                lon1=takeoff_point[1]
                original_route_dict["Latitude"].append(lat1)
                original_route_dict["Longitude"].append(lon1)
                original_route_dict["Point"].append("Takeoff") 
            else:
                df=clustered_waypoints[clustered_waypoints["Point"]==original_route[i]].reset_index(drop=True)
                lat1=df.iloc[0]["Latitude"]
                lon1=df.iloc[0]["Longitude"]
                waypoint_name=df.iloc[0]["Point"]

                original_route_dict["Latitude"].append(lat1)
                original_route_dict["Longitude"].append(lon1)
                original_route_dict["Point"].append(waypoint_name) 

            #get the position of the next point on the route (i+1)
            if i+1==len(original_route)-1:
                lat2=landing_point[0]
                lon2=landing_point[1]

            else:
                df=clustered_waypoints[clustered_waypoints["Point"]==original_route[i+1]].reset_index(drop=True)
                lat2=df.iloc[0]["Latitude"]
                lon2=df.iloc[0]["Longitude"]

            #obtain the bearing of the vector between the two points
            geod = Geodesic.WGS84.Inverse(lat1, lon1, lat2, lon2)
            bearing = geod['azi1']

            #obtain the left bound as -90 from the bearing and right bound as +90 from the bearing
            left_bound= distance(kilometers=max_distance).destination((lat1,lon1),self.CalculateBearing(bearing,-90))
            left_bounds.append((left_bound.longitude,left_bound.latitude))
            right_bound=distance(kilometers=max_distance).destination((lat1,lon1),self.CalculateBearing(bearing,+90))
            right_bounds.append((right_bound.longitude,right_bound.latitude))

        #store landing info
        original_route_dict["Latitude"].append(landing_point[0])
        original_route_dict["Longitude"].append(landing_point[1])
        original_route_dict["Point"].append("Landing")

        # convert to dataframe
        original_route_df=pd.DataFrame(original_route_dict)

        # consider landing point on its own and based on bearing of penultimate point
        left_bound=distance(kilometers=max_distance).destination(landing_point,self.CalculateBearing(bearing,-90))
        left_bounds.append((left_bound.longitude,left_bound.latitude))
        right_bound=distance(kilometers=max_distance).destination(landing_point,self.CalculateBearing(bearing,+90))
        right_bounds.append((right_bound.longitude,right_bound.latitude))
        right_bounds.reverse()

        #order the points of the polygon in the correct order going clockwise from the left bounds
        bounds=left_bounds+right_bounds

        #define the polygon using the geopandas library
        region_polygon = Polygon(bounds)
        polygon_df = gpd.GeoDataFrame(geometry=[region_polygon])

        #initialise the final waypoints database
        final_waypoints=clustered_waypoints.reset_index(drop=True)

        #iterate through waypoints database and check if waypoint is in the polygon
        for i in range(len(final_waypoints)):
            lat=clustered_waypoints.iloc[i]["Latitude"]
            lon=clustered_waypoints.iloc[i]["Longitude"]
            point_to_check = Point(lon, lat)
            is_within_region = polygon_df.contains(point_to_check).values[0]
            if is_within_region==False:
                #drop row if not in polygon
                final_waypoints=final_waypoints.drop(i)

        #plot the polygon and waypoints in polygonmap
        polygonmap=folium.Map(location=[55, -0.633950],zoom_start=6)

        #plot the original route in black
        for i in range(1,len(original_route_df)-1):

            lat=original_route_df.iloc[i]["Latitude"]
            lon=original_route_df.iloc[i]["Longitude"]
            popup=f"""
                <div style="font-size: 15px;">
                    {original_route_df.iloc[i]["Point"]}
                </div>
            """
            folium.CircleMarker(location=[lat,lon],radius=3,color="black",fill=True,fill_color="black",popup=popup).add_to(polygonmap)

        #line of original route
        folium.PolyLine(locations=original_route_df[["Latitude","Longitude"]].values,color="black",weight=2).add_to(polygonmap)

        final_waypoints=final_waypoints.reset_index(drop=True)

        #plot the bounds in red
        for i in range(len(bounds)):
            folium.CircleMarker(location=(bounds[i][1], bounds[i][0]),radius=3,color="red",fill=True,fill_color="black",popup=f"{i+1}").add_to(polygonmap)
        temp=np.flip(np.array(bounds))

        #return to starting point and close the polygon
        temp=np.vstack((temp,temp[0,:]))

        #polyline of polygon
        folium.PolyLine(locations=temp,color="red",weight=2).add_to(polygonmap)

        #plot waypoints
        for i in range(len(final_waypoints)):
            lat=final_waypoints.iloc[i]["Latitude"]
            lon=final_waypoints.iloc[i]["Longitude"]
            popup=f"""<div style="font-size: 15px;">
                    {final_waypoints.iloc[i]["Point"]}
                </div>
            """
            folium.CircleMarker(location=[lat,lon],radius=3,color="black",fill=True,fill_color="black",popup=popup).add_to(polygonmap)

        #plot take off and landing
        folium.Marker(
                location=landing_point,  
                popup=f"""
                <div style="font-size: 20px; font-weight: bold;">
                    Arrival
                </div>
            """,  
                icon=folium.CustomIcon("airport_arrival.png", icon_size=(30, 30))  
            ).add_to(polygonmap)
        
        folium.Marker(
                location=takeoff_point,  
                popup=f"""
                <div style="font-size: 20px; font-weight: bold;">
                    Departure
                </div>
            """,  
                icon=folium.CustomIcon("airport_departure.png", icon_size=(30, 30))  
            ).add_to(polygonmap)

        #store variables
        self.original_route=original_route_df
        self.takeoff_point=takeoff_point
        self.landing_point=landing_point
        self.final_waypoints=final_waypoints

        return self.original_route, self.final_waypoints, polygonmap

    '''
    This function generate a delaunay mesh from the final waypoints dataframe obtained above
    The function returns a folium map object containing the mesh for visualisation
    '''
    def GenerateDelaunayMesh(self):
        
        #store all points in an array
        all_points=np.vstack((self.final_waypoints[["Latitude","Longitude"]].values,np.array(self.takeoff_point),np.array(self.landing_point)))

        #generate mesh
        self.tri = Delaunay(all_points)

        # initialise folium map
        mesh_map=folium.Map()

        #loop through all segments of the mesh
        for indices in self.tri.simplices:

            #get the coordinates of the three points in a triangle
            point1 = tuple(all_points[indices[0]])
            point2 = tuple(all_points[indices[1]])
            point3=tuple(all_points[indices[2]])

            #create a linestring object for each connection line
            segment1= LineString([point1, point2])
            segment2= LineString([point2, point3])
            segment3= LineString([point1, point3])

            #create folium PolyLine objects for the connection lines
            weight=2
            folium.PolyLine(locations=list(segment1.coords), color='red',weight=weight).add_to(mesh_map)
            folium.PolyLine(locations=list(segment2.coords), color='red',weight=weight).add_to(mesh_map)
            folium.PolyLine(locations=list(segment3.coords), color='red',weight=weight).add_to(mesh_map)
            radius=2
            
            #plot the three points
            folium.CircleMarker(location=point1,radius=radius,color="black",fill=True,fill_color="black").add_to(mesh_map)
            folium.CircleMarker(location=point2,radius=radius,color="black",fill=True,fill_color="black").add_to(mesh_map)
            folium.CircleMarker(location=point3,radius=radius,color="black",fill=True,fill_color="black").add_to(mesh_map)

        return mesh_map

    ''' 
    Function which creates a graph object to be used in the Djikstra method. 
    The graph object contains the weight of each connection line
    
    Takes in a precipitation field of size n by 3. first column is lat, second column is lon, third column is p
    where p is the normalised precipitation intensity such that p<=0<=1
    '''    
    def CreateGraph(self,precipitation_matrix):
        
        #normalise the precipitation intensity in case it hasn't been done
        precipitation_matrix[:,2]=precipitation_matrix[:,2]/np.max(precipitation_matrix[:,2])

        #store it as a class attribute
        self.precipitation_matrix=precipitation_matrix

        #initialise graph
        graph = {}

        #store all the points of the mesh
        all_points=np.vstack((self.final_waypoints[["Latitude","Longitude"]].values,np.array(self.takeoff_point),np.array(self.landing_point)))

        #loop through each connection line
        for indices in self.tri.simplices:
            for i in range(3):
                point1 = tuple(all_points[indices[i]])
                for j in range(i+1, 3):
                    point2 = tuple(all_points[indices[j]])
                    if point1 not in graph:
                        graph[point1] = {}
                    if point2 not in graph:
                        graph[point2] = {}

                    #assign the weight for connection line between points 1 and 2
                    graph[point1][point2] = self.setWeight(point1,point2,precipitation_matrix)
                    graph[point2][point1] = graph[point1][point2]
        
        #store graph as attribute
        self.graph=graph

    #function to calculate the weight of the connection line
    def setWeight(self,point1,point2,precipitation_matrix):

        #get the geodesic distance of the connection line
        dist=geodesic(point1, point2).kilometers

        #define delta d for discretisation
        delta=5 #km
        
        #obtain number of points required to obtain the delta d value
        n=(dist // delta)-1

        #get the points along the connection line
        geo=Geod(ellps='WGS84')
        line=np.array(geo.npts(point1[1], point1[0], point2[1], point2[0], n))

        #add beginning and end points
        line= np.insert(line, 0,np.array([point1[1],point1[0]]), axis=0)
        line= np.insert(line, len(line),np.array([point1[1],point1[0]]), axis=0)
        
        #initialise the weight of the connection line 
        WEIGHT=[]
        for point in line:
            weight=0
            for pixel in precipitation_matrix:

                #obtain geodesic distance between precipitation point and point on connection line
                             #lon   ,  lat
                d=geodesic((point[1],point[0]), pixel[0:2]).kilometers

                # increment by contribution of the precipitation point
                weight+=self.WeightFormula(pixel[2],d)

            #if the weight of a point is low, consider its contribution to be zero
            if weight>=1e-2:
                WEIGHT.append(weight)
            else:
                WEIGHT.append(0)
        # weight of entire connection line is the average of the individual contributions
        #if the weight of a line, consider its precipitation contribution to be zero
        if np.mean(WEIGHT)>=0.05:
        
            WEIGHT=np.mean(WEIGHT)
        else:
            WEIGHT=0
        
        #return the total weight of the line, adding distance in the formula
        return (1+WEIGHT)*dist
    
    # formula for the contribution of a pixel on a point along a connection line
    def WeightFormula(self,p,d):
        return p*np.exp(-d/2)

    # function which calculates the new bearing value obtained from heading+angle. 
    # used because sometimes the bearing obtained is negative or greater than 360
    def CalculateBearing(self,heading,angle):
        new_bearing=heading+angle #angle needs to be negative if going anticlockwise
        if new_bearing <0 :
            new_bearing=new_bearing+360
        return new_bearing

    # dijkstra's algorithm as a function, no input is required
    def Dijkstra(self):
        # Priority queue to store vertices and their distances
        start=self.takeoff_point
        end=self.landing_point
        graph=self.graph
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
        
        #same process as in the ClusterWaypoints method
        tolerance=0.000001
        optimised_path=pd.DataFrame()
        final_waypoints=self.final_waypoints
        for waypoint in shortest_path:
            df = final_waypoints[
            (abs(final_waypoints["Latitude"] - waypoint[0]) < tolerance) &
            (abs(final_waypoints["Longitude"] - waypoint[1]) < tolerance)
                ]
            optimised_path=pd.concat([optimised_path,df],axis=0)
        
        #add take off and landing to the dataframe containing the optimised path
        takeoff_point_dict={
            "Latitude": self.takeoff_point[0],
            "Longitude": self.takeoff_point[1],
            "Point": "Takeoff"
        }
        landing_point_dict={
            "Latitude": self.landing_point[0],
            "Longitude": self.landing_point[1],
            "Point": "Landing"
        }
        optimised_path = pd.concat([pd.DataFrame(takeoff_point_dict, index=[0]), optimised_path], ignore_index=True)
        optimised_path = pd.concat([optimised_path,pd.DataFrame(landing_point_dict, index=[0])], ignore_index=True)

        self.optimised_path=optimised_path
    
    # function which generate a map showing the optimised route against the original route 
    # along with the weather avoided (if successful)
    #no inputs but show_mesh (True/False) and mesh_color are optional inputs which are self explanatory. 
    # format for mesh_color is a pre-defined color in folium or a hex code
    def GenerateMapOriginalVsOptimal(self,**kwargs):

        #default value
        show_mesh=True
        if "show_mesh" in kwargs.keys() and kwargs.get('show_mesh')==True:

            if "mesh_color" in kwargs.keys():
                mesh_color = kwargs.get('mesh_color')
            else:
                mesh_color="grey"
        elif "show_mesh" in kwargs.keys() and kwargs.get('show_mesh')==False:
            show_mesh=False
        
        # same process as in earlier functions
        map_optim_vs_original=folium.Map(location=[55, -0.633950],zoom_start=6)
        all_points=np.vstack((self.final_waypoints[["Latitude","Longitude"]].values,np.array(self.takeoff_point),np.array(self.landing_point)))
        if show_mesh==True:
            for indices in self.tri.simplices:

                point1 = tuple(all_points[indices[0]])
                point2 = tuple(all_points[indices[1]])
                point3=tuple(all_points[indices[2]])

                segment1= LineString([point1, point2])
                segment2= LineString([point2, point3])
                segment3= LineString([point1, point3])
                
                weight=2
                folium.PolyLine(locations=list(segment1.coords), color=mesh_color,weight=weight).add_to(map_optim_vs_original)
                folium.PolyLine(locations=list(segment2.coords), color=mesh_color,weight=weight).add_to(map_optim_vs_original)
                folium.PolyLine(locations=list(segment3.coords), color=mesh_color,weight=weight).add_to(map_optim_vs_original)
                radius=2
                folium.CircleMarker(location=point1,radius=radius,color="black",fill=True,fill_color="black").add_to(map_optim_vs_original)
                folium.CircleMarker(location=point2,radius=radius,color="black",fill=True,fill_color="black").add_to(map_optim_vs_original)
                folium.CircleMarker(location=point3,radius=radius,color="black",fill=True,fill_color="black").add_to(map_optim_vs_original)

        original_route=self.original_route
        optimised_path=self.optimised_path
        for i in range(1,len(original_route)-1):

            lat=original_route.iloc[i]["Latitude"]
            lon=original_route.iloc[i]["Longitude"]
            popup=f"""
            <div style="font-size: 15px;">
                {original_route.iloc[i]["Point"]}
            </div>
            """
            folium.CircleMarker(location=[lat,lon],radius=5,color="black",fill=True,fill_color="black",popup=popup).add_to(map_optim_vs_original)
    
        folium.PolyLine(locations=original_route[["Latitude","Longitude"]].values,color="black").add_to(map_optim_vs_original)

        for i in range(len(optimised_path)):

            lat=optimised_path.iloc[i]["Latitude"]
            lon=optimised_path.iloc[i]["Longitude"]
 
            if i==0 or i==len(optimised_path)-1:
                popup=f"""
                <div style="font-size: 20px; font-weight: bold;">
                   {optimised_path.iloc[i]["Point"]}
                </div>
                """
                if i==0:
                    folium.Marker(location=[lat,lon],popup=popup,icon=folium.CustomIcon("airport_departure.png", icon_size=(30, 30))  ).add_to(map_optim_vs_original)
                elif i==len(optimised_path)-1:
                    folium.Marker(location=[lat,lon],popup=popup,icon=folium.CustomIcon("airport_arrival.png", icon_size=(30, 30))  ).add_to(map_optim_vs_original)
            else:
                popup=f"""
                <div style="font-size: 20px;">
                   {optimised_path.iloc[i]["Point"]}
                </div>
                """
                folium.CircleMarker(location=[lat,lon],radius=5,color="red",fill=True,fill_color="black",popup=optimised_path.iloc[i]["Point"]).add_to(map_optim_vs_original)

        folium.PolyLine(locations=optimised_path[["Latitude","Longitude"]].values,color="red").add_to(map_optim_vs_original)
        HeatMap(self.precipitation_matrix).add_to(map_optim_vs_original)

        #return the map
        return map_optim_vs_original