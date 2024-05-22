import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import math
import os
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from FlightRadar24 import FlightRadar24API
from geopy.distance import distance, geodesic
from geopy.point import Point
import folium
import requests
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
import pygeodesy
from pyproj import Geod
import pandas as pd
'''
   +---------------------------------------------------+
   | Code Developed by Jade Nassif                     |
   | Aerospace Computational Engineering MSc 2023-2024 |
   | Cranfield University                              |
   | jade.nassif.385@cranfield.ac.uk                   |
   +---------------------------------------------------+ 
'''

'''
The following code contains the following classes:

1. CentralDataHandler which acts as a parent class to classes 2 
and 3. It contains the functions common to the two classes

2. FlightData which takes in flight details and retrieves realtime
flight info through the FlightRadar24API. The parent class' functions
are then accessed to execute the different tasks

3. FlightPlanning which takes in a pre-defined trail/route from a .csv file.
The file contains the following information: 
lat,lon,alt,timestamp,heading,ground speed
The trails are obtained through the FlightData class and stored locally

4. NearbyMETARInfo which returns the METAR reports of airports within a 
certain radius of the current aircraft's position 
'''
# -----------------------------------
# Definition of the parent class
class CentralDataHandler:

    #constructor. takes in radius of interest around the aircraft
    def __init__(self, radius):
        self.radius=radius # radius in kilometers, same throughout this code
    
    '''
    function which generates a structured square grid based on :
    - the current position of the aircraft
    - the view chosen (true north or heading) 

    inputs: 
    - lat : latitude of the aircraft
    - lon : longitude of the aircraft
    - heading : heading of the aircraft in degrees clockwise from true north
    - n : number of points in each direction, grid is n by n
    - view : "true north" or "heading"
    '''
    def GenerateMesh(self,lat,lon,heading,n,view):

        # X-Y grid used for plotting later (projected square)
        self.X,self.Y=np.meshgrid(np.linspace(0,2*self.radius,n),np.linspace(0,2*self.radius,n))

        #check input is correct
        if view!="true north" and view!="heading":
            raise ValueError("variable 'view' can either be 'true north' or 'heading'")
        
        # get points e,f,g and h based on the current position of the aircraft
        #(see report for definition of e,f,g,h)
        e,f,g,h=self.CalculateRadiusPoints(lat,lon,heading,view)
        
        # calculate the intersection a,b,c and d to get the boundary points of the mesh
        if view=="true north":
            a=self.CalculateIntersection(e.latitude,e.longitude,h.latitude,h.longitude,180,270)
            b=self.CalculateIntersection(g.latitude,g.longitude,h.latitude,h.longitude,180,90)
            c=self.CalculateIntersection(e.latitude,e.longitude,f.latitude,f.longitude,0,270)
            d=self.CalculateIntersection(g.latitude,g.longitude,f.latitude,f.longitude,0,90)
        elif view=="heading":
            a=self.CalculateIntersection(e.latitude,e.longitude,h.latitude,h.longitude,self.CalculateBearing(heading,180),self.CalculateBearing(heading,270))
            b=self.CalculateIntersection(g.latitude,g.longitude,h.latitude,h.longitude,self.CalculateBearing(heading,180),self.CalculateBearing(heading,90))
            c=self.CalculateIntersection(e.latitude,e.longitude,f.latitude,f.longitude,heading,self.CalculateBearing(heading,270))
            d=self.CalculateIntersection(g.latitude,g.longitude,f.latitude,f.longitude,heading,self.CalculateBearing(heading,90))
        
        #initialise Geod object
        # use WGS84 ellipsoid model for best geodesical accuracy
        geod=Geod(ellps='WGS84')

        #get n-2 equally spaced points along the sides of the square 
        a_to_b=np.array(geod.npts(a.longitude, a.latitude, b.longitude, b.latitude, n-2))
        c_to_d=np.array(geod.npts(c.longitude, c.latitude, d.longitude, d.latitude, n-2))
        a_to_c=np.array(geod.npts(a.longitude, a.latitude, c.longitude, c.latitude, n-2))
        b_to_d=np.array(geod.npts(b.longitude, b.latitude, d.longitude, d.latitude, n-2))

        #initialise 
        mesh_latitude=np.zeros((n,n))
        mesh_longitude=np.zeros((n,n))

        #enter a, b, c and d (boundaries)
        mesh_latitude[0,0]=a.latitude
        mesh_longitude[0,0]=a.longitude
        mesh_latitude[0,-1]=c.latitude
        mesh_longitude[0,-1]=c.longitude
        mesh_latitude[-1,0]=b.latitude
        mesh_longitude[-1,0]=b.longitude
        mesh_latitude[-1,-1]=d.latitude
        mesh_longitude[-1,-1]=d.longitude

        #store sides of the square obtained above
        # a to c
        mesh_latitude[0,1:-1]=np.transpose(a_to_c[:,1])
        mesh_longitude[0,1:-1]=np.transpose(a_to_c[:,0])
        # b to d
        mesh_latitude[-1,1:-1]=np.transpose(b_to_d[:,1])
        mesh_longitude[-1,1:-1]=np.transpose(b_to_d[:,0])
        # a to b
        mesh_latitude[1:-1,0]=a_to_b[:,1]
        mesh_longitude[1:-1,0]=a_to_b[:,0]
        #c to d
        mesh_latitude[1:-1,-1]=c_to_d[:,1]
        mesh_longitude[1:-1,-1]=c_to_d[:,0]

        #loop through i and store the curve each time 
        for i in range(1,n-1):
            
            top_lat=mesh_latitude[i,-1]
            top_lon=mesh_longitude[i,-1]
            bottom_lat=mesh_latitude[i,0]
            bottom_lon=mesh_longitude[i,0]

            bottom_to_top=np.array(geod.npts(bottom_lon, bottom_lat, top_lon, top_lat, n-2))

            mesh_latitude[i,1:-1]=np.transpose(bottom_to_top[:,1])
            mesh_longitude[i,1:-1]=np.transpose(bottom_to_top[:,0])
        #flatten longitude and latitude mesh into a single array
        self.lats_longs=np.transpose(np.array([mesh_latitude.flatten(),mesh_longitude.flatten()]))

    # function which calculates the positions of the points e,f,g and h based on the radius and view chosen
    # used in GenerateMesh function
    def CalculateRadiusPoints(self,lat,lon,heading,view):

        #define aircraft location
        aircraft_location=Point(lat,lon)
        #calculate the bearing of the points e,f,g and h from the aircraft's location
        if view=="heading":
            bearing_e=self.CalculateBearing(heading,-90)
            bearing_f=heading
            bearing_g=self.CalculateBearing(heading,+90)
            bearing_h=self.CalculateBearing(heading,180)
        elif view=="true north":
            bearing_e=270
            bearing_f=0
            bearing_g=90
            bearing_h=180
        
        #obtain the position of the four points
        point_e=distance(kilometers=self.radius).destination(aircraft_location,bearing_e)
        point_f=distance(kilometers=self.radius).destination(aircraft_location,bearing_f)
        point_g=distance(kilometers=self.radius).destination(aircraft_location,bearing_g)
        point_h=distance(kilometers=self.radius).destination(aircraft_location,bearing_h)

        return point_e,point_f,point_g,point_h

    #function which calculates the intersection between two points and their headings
    # uses Great Circle WGS84 model 
    def CalculateIntersection(self,lat1,lon1,lat2,lon2,hd1,hd2):
        #hd refers to heading
        inter = pygeodesy.formy.intersection2(lat1, lon1, hd1, lat2, lon2, hd2,datum=pygeodesy.Datums.WGS84)
        intersection=Point(inter[0],inter[1])
        return intersection
    
    #function which gets the wind speed data at every point of the mesh
    def GetWindData(self):

        # initialise the wind_deg and wind_speed vectors
        self.wind_deg=np.zeros((len(self.lats_longs),1))
        self.wind_speed=np.zeros((len(self.lats_longs),1))

        # personal api key for OpenWeatherMap
        api_key = "15c250dbc504a1c9e7547e27ea196a2c"
        
        #iterate through the coordinates
        for i in range(0,len(self.lats_longs)):
        
            url = f"https://api.openweathermap.org/data/2.5/weather?lat={self.lats_longs[i,0]}&lon={self.lats_longs[i,1]}&appid={api_key}"

            #request data
            data=requests.get(url).json()

            #extract wind speed and direction
            wind_speed=data["wind"]["speed"]
            wind_deg=data["wind"]["deg"]

            self.wind_deg[i,0]=wind_deg
            self.wind_speed[i,0]=wind_speed

        #reshape the vectors to match the size of the mesh
        new_size=int(math.sqrt(len(self.lats_longs)))
        self.wind_deg=np.reshape(self.wind_deg,(new_size,new_size))
        self.wind_speed=np.reshape(self.wind_speed,(new_size,new_size))
        
    # function which calculates the new bearing value obtained from heading+angle. 
    # used because sometimes the bearing obtained is negative or greater than 360
    def CalculateBearing(self,heading,angle):

        new_bearing=heading+angle #angle needs to be negative if going anticlockwise
        if new_bearing <0 :
            new_bearing=new_bearing+360
        return new_bearing
    
    '''
    function which returns a streamplot figure object used in the UI to represent the wind

    inputs: 
    - X and Y are the meshgrid objects
    - U and V are the X and Y components of the wind in the local x,y coordinate system (see report for clarification)
    - plot_mode refers to whether the gradient of the wind or its speed should be used to color the streamplot. (can take on
    "gradient" or "speed")
    - width and height are the width and height of the figure
    - view is the same as previously defined in the code
    '''
    def PlotWind(self,X,Y,U,V,plot_mode,width,height,view):

        #get x and y in linspace form
        # x=np.unique(X.flatten())
        # y=np.unique(Y.flatten())

        # calculate the color gradient matrix based on the plot mode chosen
        if plot_mode=="gradient":

            #calculate the gradient of the wind in each direction and then magnitude of gradient
            S=np.sqrt(U**2+V**2)
            dS_dx,dS_dy=np.gradient(S)


            color = np.sqrt(dS_dx**2+dS_dy**2)
        elif plot_mode=="speed":
            color=np.sqrt(U**2+V**2)
        else:
            raise ValueError("the input 'mode' is either 'gradient' or 'speed'")

        #initialise the figure
        fig=Figure(figsize=(width,height))
        ax = fig.add_subplot(111)
        ax.streamplot(X, Y, U, V, density=1,color=color,broken_streamlines=True)
        ax.axis("off")

        #parameters defining the arrow head of the aircraft
        frac=0.05
        frac2=0.025

        #orient the arrow based on the view chosen
        if view=="true north":

            x_arrow,y_arrow=self.RotateArrow(frac,frac2)
            xy=np.array([[0.5],[0.9]])

        elif view=="heading":
            x_arrow=np.array([-frac,0,frac,0])+0.5
            y_arrow=np.array([-frac,-frac2,-frac,frac])+0.5
            xy=self.RotatePoint(0,0.15,self.current_info["heading"])
            xy[0,0]+=0.5
            xy[1,0]+=0.75

        #add the heading values below the arrowhead
        ax.text(0.5, 0.4, f"{int(self.current_info['heading'])}", ha='center', va='center', transform=ax.transAxes,fontsize=12,color='white',bbox=dict(facecolor='black', alpha=0.5))
        ax.fill(x_arrow,y_arrow,color="black",transform=ax.transAxes)
        #add the north direction arrow
        ax.annotate('N', xy=(xy[0,0]*2*self.radius, xy[1,0]*2*self.radius),xytext=(0.5*2*self.radius, 0.75*2*self.radius),arrowprops=dict(facecolor='black', arrowstyle='->',linewidth=4,mutation_scale=15),ha='center',va='center',color='white',fontsize=15,bbox=dict(facecolor='black', alpha=0.5))

        ax.set_aspect('equal')

        return fig
    
    # function which rotates a point about the point 0,0 by an angle (transformation matrix)
    def RotatePoint(self,x,y,angle_deg):

        #convert to radians
        angle=angle_deg*np.pi/180
        #rotation matrix
        R = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])
        xy = np.array([[x], [y]])
        xy_new=np.dot(R,xy)
        return xy_new

    #function which rotates the arrow head defining the aircraft based on the heading
    def RotateArrow(self,frac,frac2):

        #x base x y of the arrow at 0,0
        x=np.transpose(np.array([-frac,0,frac,0]))
        y=np.transpose(np.array([-frac,-frac2,-frac,frac]))

        #calculate angle required to rotate about
        angle=2*np.pi -self.current_info["heading"]*np.pi/180
        #rotation matrix
        R = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])
        xy=np.column_stack((x,y))
        for i in range(0,len(xy)):
            xy[i,:]=np.dot(R,xy[i,:])
        x=np.transpose(xy[:,0])+0.5
        y=np.transpose(xy[:,1])+0.5

        return x,y
    
    #function which takes in a trail defined in a .csv file and returns a df
    def TrailtoDf(self,filename):
        trail_df=pd.read_csv(filename)
        return trail_df

#FlightData Class definition
class FlightData(CentralDataHandler):

    '''
    Constructor. Inputs:
    - radius same as in CentralDataHandler, in km
    - flight number of the target flight
    - aircraft type of the target flight in ICAO format (eg. B737)
    - airline ICAO of the target flight
    '''
    def __init__(self,radius,flight_number_target,aircraft_type,airline_ICAO):

        super().__init__(radius)
        self.flight_number_target=flight_number_target
        self.aircraft_type=aircraft_type
        self.airline_ICAO=airline_ICAO

    #function which obtains the current flight data using the flightradar24 API
    def GetCurrentFlightInfo(self):

        #initialise the api object
        fr_api = FlightRadar24API()

        #get zones and initialise the flights object
        zones=list((fr_api.get_zones()).keys())
        flights=[]

        #while loop until flight is found
        i=0
        correct_flight=False
        while i<len(zones) and correct_flight==False:
            #go through the zones
            zone=fr_api.get_zones()[zones[i]]
            bounds = fr_api.get_bounds(zone)
            #obtain the flights in that zone matching the criteria
            flights = fr_api.get_flights(aircraft_type = self.aircraft_type,airline = self.airline_ICAO,bounds = bounds)
            #if flights is not empty
            if flights!=[]:
                for flightt in flights:
                    #check if any of the flights have the correct flight number
                    if flightt.number==self.flight_number_target:
                        flight=flightt
                        correct_flight=True
            i+=1

        if correct_flight==False:
            raise IOError("The request flight has not been found")
        
        #initialise dictionary containing the current information
        self.current_info={}

        #store the current information
        self.current_info["lat"]=flight.latitude
        self.current_info["lon"]=flight.longitude
        self.current_info["alt"]=flight.altitude
        self.current_info["timestamp"]=flight.time #need to convert to time format
        self.current_info["heading"]=flight.heading
        self.current_info["ground speed"]=flight.ground_speed
        #store the trail of the aircraft so far
        self.trail=fr_api.get_flight_details(flight)["trail"]
    
    #function which write the trail to a csv file
    def TrailtoCSV(self):

        filename=f"flight_{self.flight_number_target}.csv"
        i=1
        #check if the file already exists
        while os.path.exists(filename)==True:
            filename=f"flight_{self.flight_number_target}_{i}.csv"
            i=i+1
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = list(self.current_info.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(0,len(self.trail)):
                writer.writerow({
                    "lat": self.trail[i]["lat"],
                    "lon": self.trail[i]["lng"],
                    "alt": self.trail[i]["alt"],
                    "timestamp": self.trail[i]["ts"],
                    "heading": self.trail[i]["hd"],
                    "ground speed": self.trail[i]["spd"],
                })

    # function which uses the GenerateMesh method defined in the parent class to generate the grid
    def GenerateMesh(self,n,mode):
        super().GenerateMesh(self.current_info["lat"],self.current_info["lon"],self.current_info["heading"],n,mode)

    # function which uses the GetWindData method defined in the parent class to get the wind data
    def GetWindData(self):
        super().GetWindData()

    # function which uses the PlotWind method defined in the parent class to plot the wind data
    # U and V are first calculated based on the view chosen by calculating the wind direction in the
    # local x,y axes
    def PlotWind(self,plot_mode,width,height,view):

        if view=="true north":
            angles=(90-self.wind_deg)*np.pi/180
        elif view=="heading":
            angles=(90-self.wind_deg+self.current_info["heading"])*np.pi/180

        U=self.wind_speed*np.cos(angles)
        V=self.wind_speed*np.sin(angles)

        fig=super().PlotWind(self.X,self.Y,U,V,plot_mode,width,height,view)
        return fig
    
    #write the current info in a simple text file
    def WriteCurrentInfo(self,text_filename):
        filename=text_filename
        with open(filename,'w') as file:
            for key,value in self.current_info.items():
                file.write(f"{key}: {value}\n")
            file.write(f"radius: {self.radius}\n")

# FlightPlanning Class definition
class FlightPlanning(CentralDataHandler):

    #constructor, takes in the trail .csv files as a pandas dataframe
    def __init__(self,radius,trail_df):
        super().__init__(radius)

        self.trail=trail_df

    # same as in FlightData. 
    # lat and lon and be obtained by an interpolation in the dataframe 
    # this was implemented directly in the UI code as it depends on the time spanned since pressing a button
    def GenerateMesh(self,lat,lon,heading,n,mode):
        super().GenerateMesh(lat,lon,heading,n,mode)
        self.current_info={
            "heading":heading
        }
    
    # same as in FlightData. 
    def GetWindData(self):
        super().GetWindData()
    
    # same as in FlightData. 
    def PlotWind(self,plot_mode,width,height,view):
        if view=="true north":
            angles=(90-self.wind_deg)*np.pi/180
        elif view=="heading":
            angles=(90-self.wind_deg+self.current_info["heading"])*np.pi/180
        U=self.wind_speed*np.cos(angles)
        V=self.wind_speed*np.sin(angles)
        fig=super().PlotWind(self.X,self.Y,U,V,plot_mode,width,height,view)
        return fig

# NearbyMETARInfo class definition
class NearbyMETARInfo():
    # constructor
    # radius of interest in km
    # lat and lon of current location
    def __init__(self,radius,lat,lon):

        self.radius=radius
        
        self.lat=lat
        self.lon=lon
        #url defining the API request. radius is entered in miles in the request hence the /1.609
        self.url = f"https://api.checkwx.com/metar/lat/{self.lat}/lon/{self.lon}/radius/{int(self.radius/1.609)}/decoded"

    # request the data from the API
    def GetNearbyMETARInfo(self):

        response = requests.request("GET", self.url, headers={'X-API-Key': '16adb1b8a0f54fb1a8c448ba93'})

        #attribute which stores all the data from the API request
        self.all_metar_data=(response.json())["data"]

        ''' INSTRUCTIONS ON HOW TO ACCESS SOME OF THE FIELDS BELOW'''
        # self.all_metar_data=metar_data["data"]
        #to access the following:
        # ICAO: self.all_metar_data[i]["icao"]
        # General cloud status (eg overcast) and cloud height/ceiling : self.all_metar_data[i]["clouds"][0]["text"] or ['meters'] for ceiling
        # general condition of weather self.all_metar_data[i]["conditions"][0]["code"] or "text"
        # dewpoint self.all_metar_data[i]["dewpoint"]["celsius"]
        #humidity self.all_metar_data[i]["humidity"]["percent"]
        #distance away from current location self.all_metar_data[i]["meters"]
        #coordinates of station self.all_metar_data[i]["station"]["geometry"]["coordinates"][0] for longitude and [1] for latitude
        # temperature  self.all_metar_data[i]["temperature"]["celsius"]
        # raw metar info as a string self.all_metar_data[i]["raw_text"]
        # visibility self.all_metar_data[i]["visibility"]["meters"]
        # wind self.all_metar_data[i]["wind"]["degrees"] or ["speed_kts"] or ["gust_kts"]
    
    # functions which extracts the raw METAR reports from the response
    # output is a list where every entry is a METAR report, in order from
    # the nearest airport to the furthest
    def GetRawMETARList(self):
        
        self.raw_metar_text=[]
        
        for airport in self.all_metar_data:
            self.raw_metar_text.append(airport["raw_text"])

        return self.raw_metar_text
    
    # obtain the coordinates of each METAR station. same format as the above function
    def GetMETARCoordinates(self):

        self.metar_coordinates=[]

        for airport in self.all_metar_data:
            self.metar_coordinates.append(airport["station"]["geometry"]["coordinates"])
        
        return self.metar_coordinates
    
    #return the wind properties as a list, same format as above
    def GetWindProperties(self):

        self.wind_direction=[]
        self.wind_speed=[]

        #if the wind information is available, append the value otherwise append None
        for airport in self.all_metar_data:

            try:
                self.wind_direction.append(airport["wind"]["degrees"])
            except:
                self.wind_direction.append(None) 
            
            try:
                self.wind_speed.append(airport["wind"]["speed_kts"])
            except:
                self.wind_speed.append(None)
        
        return self.wind_direction,self.wind_speed