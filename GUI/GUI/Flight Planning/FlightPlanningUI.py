from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap
from PyQt5 import uic
from PyQt5.QtCore import QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView
import folium
import numpy as np
import pandas as pd
from class_definition import OptimisePath
import re
import warnings
import random
warnings.filterwarnings("ignore")
import os
'''
   +---------------------------------------------------+
   | Code Developed by Jade Nassif                     |
   | Aerospace Computational Engineering MSc 2023-2024 |
   | Cranfield University                              |
   | jade.nassif.385@cranfield.ac.uk                   |
   +---------------------------------------------------+ 
'''

''' 
This code makes use of the OptimisePath Class defined in class_definition.py to 
define the FlightPlannerGUI in the FlightPlanner Class. The program follows the same steps as those defined in the 
class definition. It allows the pilot to select a departure and arrival airport, as well 
as a runway. It then displays the Great Circle line between the departure and arrival airports,
which helps the pilot interatively select waypoints to define their route. It can then be entered 
in the following format WAYPT1-WAYPT2-...-LASTWAYPOINT and this displays a travel corridor spanning 
a default values of 50 km from the route on both sides. the pilot then has the option of optimising 
the route for the weather. 

The precipitation field is an input into the constructor and an example on how to generate 
a fake field based on a Gaussian Normal Distribution is shown after the class definition
'''

class FlightPlanner(QMainWindow):

    #constructor, input is the precipitation field 
    # np array with three columns, first column is lat, second is lon and third is normalised
    # precipitation intensity (0<=p<=1)
    def __init__(self,precipitation_matrix):

        #load the precipitation field
        self.precipitation_matrix=precipitation_matrix

        #initialise the UI

        super(FlightPlanner,self).__init__()

        #load the UI file
        uic.loadUi("FlightPlanningUI_Final.ui",self)
        self.show()
        self.setWindowTitle("Airwaze Flight Planner")

        #get database of airports
        airports_df=pd.read_csv("Airports_Pool.csv",delimiter=';')
        self.airports_df = airports_df.dropna()

        # initialise UI
        self.InitialiseButtons()

        #conect the runways to the airport selection dropdowns for departure
        departure_airport=self.flight_details_menu.itemAt(2).widget()
        departure_airport.currentIndexChanged.connect(self.UpdateDepartureRunway)

        #conect the runways to the airport selection dropdowns for arrival
        arrival_airport=self.flight_details_menu.itemAt(7).widget()
        arrival_airport.currentIndexChanged.connect(self.UpdateArrivalRunway)

        #initialise variable which checks if ready to submit flight details
        self.ready_to_submit=False

        #connect the submit button to the function
        submit_flight_details_button=self.flight_details_menu.itemAt(11).widget()
        submit_flight_details_button.clicked.connect(self.SubmitFlightDetails)

        #get the dataframe for waypoints
        waypoints_df=pd.read_csv("waypoints.csv",delimiter=';')
        self.waypoints_df = waypoints_df.dropna()

        #initialise variable which checks if ready to start optimisation
        self.ready_to_optim=False

        #connect enter route button to function
        original_route_menu=self.route_optimisation_menu.itemAt(1)
        submit_route_button=original_route_menu.itemAt(1).widget()
        submit_route_button.clicked.connect(self.EnterRoute)

        #connect the optimise route button to function
        self.route_entered=False
        optimise_route_button=self.route_optimisation_menu.itemAt(2).widget()
        optimise_route_button.clicked.connect(self.OptimiseForWeather)

        pixmap = QPixmap("background.jpg")

        # Set the background image to the background label (zz)
        self.zz.setPixmap(pixmap)
        self.zz.lower()

    #sets the buttons to be enabled or disabled
    def InitialiseButtons(self):

        #disable the optimisation related parts
        original_route_menu=self.route_optimisation_menu.itemAt(1)

        route_text_edit=original_route_menu.itemAt(0).widget()
        route_text_edit.setReadOnly(True)

        submit_route_button=original_route_menu.itemAt(1).widget()
        submit_route_button.setEnabled(False)

        optimise_route_button=self.route_optimisation_menu.itemAt(2).widget()
        optimise_route_button.setEnabled(False)

        #add the lists to the airport selection dropdowns
        airports_list=self.airports_df['map'].unique().tolist()
        airports_list=["Select Airport"]+airports_list

        departure_airport=self.flight_details_menu.itemAt(2).widget()
        departure_airport.addItems(airports_list)

        arrival_airport=self.flight_details_menu.itemAt(7).widget()
        arrival_airport.addItems(airports_list)
    
    # function which controls the runway dropdown for departures
    def UpdateDepartureRunway(self):

        #extract currently selected airport
        selected_airport=self.flight_details_menu.itemAt(2).widget().currentText()

        #get the runways from the database
        condition = self.airports_df['map'] == selected_airport
        runways=self.airports_df.loc[condition,"Runway Name"].tolist()
        departure_runway=self.flight_details_menu.itemAt(4).widget()

        #clear existing items and add the runways
        departure_runway.clear()
        departure_runway.addItems(runways)

    #same as above but for arrival
    def UpdateArrivalRunway(self):

        #extract currently selected airport
        selected_airport=self.flight_details_menu.itemAt(7).widget().currentText()

        #get the runways from the database
        condition = self.airports_df['map'] == selected_airport
        runways=self.airports_df.loc[condition,"Runway Name"].tolist()
        arrival_runway=self.flight_details_menu.itemAt(9).widget()

        #clear existing items and add runways
        arrival_runway.clear()
        arrival_runway.addItems(runways)
    
    # function which controls what the submit flight details button does
    def SubmitFlightDetails(self):

        #check if the flight details are valid
        self.CheckFlightDetails()

        if self.ready_to_submit:
            #disable dropdowns 
            submit_flight_details_button=self.flight_details_menu.itemAt(11).widget()
            departure_airport=self.flight_details_menu.itemAt(2).widget()
            arrival_airport=self.flight_details_menu.itemAt(7).widget()
            departure_runway=self.flight_details_menu.itemAt(4).widget()
            arrival_runway=self.flight_details_menu.itemAt(9).widget()

            submit_flight_details_button.setEnabled(False)
            departure_airport.setEnabled(False)
            arrival_airport.setEnabled(False)
            departure_runway.setEnabled(False)
            arrival_runway.setEnabled(False)

            # initialise the OptimisePath instance with the waypoints dataframe
            self.flight=OptimisePath(self.waypoints_df)
            #obtain the clustered waypoints database and the map of clustered waypoints
            _,self.clustered_waypoints_df,clustered_waypoints_map=self.flight.ClusterWaypoints()

            #get the position of the departure runway (start lat and lon)
            condition = (self.airports_df['map'] == departure_airport.currentText()) & (self.airports_df['Runway Name'] == departure_runway.currentText())
            df = self.airports_df.loc[condition]
            take_off = df[["Start Latitude", "Start Longitude"]].values
            #convert to tuple
            self.take_off=(take_off[0,0],take_off[0,1])

            #get the position of the arrival runway (end lat and lon)
            condition = (self.airports_df['map'] == arrival_airport.currentText()) & (self.airports_df['Runway Name'] == arrival_runway.currentText())
            df = self.airports_df.loc[condition]
            landing= df[["Start Latitude", "Start Longitude"]].values
            #convert to tuple
            self.landing=(landing[0,0],landing[0,1])

            #style for the pop up in the folium map for departure point
            popup_take_off = f"""
                <div style="font-size: 20px; font-weight: bold;">
                    Departure
                </div>
            """
            folium.Marker(
                location=self.take_off,  
                popup=popup_take_off,  
                icon=folium.CustomIcon("airport_departure.png", icon_size=(50, 50)) 
            ).add_to(clustered_waypoints_map)

            #style for the pop up in the folium map for arrival
            popup_landing = f"""
                <div style="font-size: 20px; font-weight: bold;">
                    Arrival
                </div>
            """
            folium.Marker(
                location=self.landing,  
                popup=popup_landing,  
                icon=folium.CustomIcon("airport_arrival.png", icon_size=(50, 50))  
            ).add_to(clustered_waypoints_map)

            #draw great circle line from take off to landing
            folium.PolyLine(locations=[self.take_off, self.landing]).add_to(clustered_waypoints_map)
            clustered_waypoints_map.save('clustered_waypoints_map.html')

            #QWebGraphics widget where map will be displayed
            map_view=self.map_layout.itemAt(0).widget()

            #process to add the map to the widget
            scene = QGraphicsScene()
            webview = QWebEngineView()
            webview.setFixedSize(map_view.width(), map_view.height())
            current_directory = os.getcwd()
            file_name = "clustered_waypoints_map.html"
            full_path = os.path.join(current_directory, file_name)
            url=QUrl.fromLocalFile(full_path)
            webview.setUrl(url) 
            scene.addWidget(webview)
            map_view.setScene(scene)

            # Renable the optimisation workflow
            original_route_menu=self.route_optimisation_menu.itemAt(1)

            route_text_edit=original_route_menu.itemAt(0).widget()
            route_text_edit.setReadOnly(False)

            submit_route_button=original_route_menu.itemAt(1).widget()
            submit_route_button.setEnabled(True)

            optimise_route_button=self.route_optimisation_menu.itemAt(2).widget()
            optimise_route_button.setEnabled(True)
    
    #function which defines what happens when the Enter Route button is clicked
    def EnterRoute(self):

        #check route entered is okay
        self.CheckOriginalRoute()
        
        if self.ready_to_optim==True:

            #obtain the map showing the flight corridor
            _,_,polygonmap=self.flight.FilterWaypoints(original_route=self.original_route,takeoff_point=self.take_off,landing_point=self.landing,maximum_distance_away_from_route=50)
            polygonmap.save("corridor.html")

            #similar process as cluster waypoints function
            map_view=self.map_layout.itemAt(0).widget()

            scene = QGraphicsScene()
            webview = QWebEngineView()
            webview.setFixedSize(map_view.width(), map_view.height())
            current_directory = os.getcwd()
            file_name = "corridor.html"
            full_path = os.path.join(current_directory, file_name)
            url=QUrl.fromLocalFile(full_path)
            webview.setUrl(url) 
            scene.addWidget(webview)
            map_view.setScene(scene)

            #set the boolean to True
            self.route_entered=True

            #disable the workflow, only optimisation button is enable
            original_route_menu=self.route_optimisation_menu.itemAt(1)
            route_text_edit=original_route_menu.itemAt(0).widget()
            route_text_edit.setReadOnly(True)

            submit_route_button=original_route_menu.itemAt(1).widget()
            submit_route_button.setEnabled(False)

    # function which defines what happens when Optimise for Weather button is clicked
    def OptimiseForWeather(self):

        if self.route_entered==True:

            #generate a Delaunay mesh 
            self.flight.GenerateDelaunayMesh()
            
            # get precipitation matrix
            precip=self.precipitation_matrix

            #normalise
            precip[:,2]/=np.max(precip[:,2])

            #create the graph object
            self.flight.CreateGraph(precipitation_matrix=precip)

            #launch the optimisation
            self.flight.Dijkstra()

            # get the map showing original and optimised routes
            map_optim_vs_original=self.flight.GenerateMapOriginalVsOptimal(show_mesh=False)
            map_optim_vs_original.save("map_optim_vs_original.html")

            #same process as before to display the map
            map_view=self.map_layout.itemAt(0).widget()

            scene = QGraphicsScene()
            webview = QWebEngineView()
            webview.setFixedSize(map_view.width(), map_view.height())
            current_directory = os.getcwd()
            file_name = "map_optim_vs_original.html"
            full_path = os.path.join(current_directory, file_name)
            url=QUrl.fromLocalFile(full_path)
            webview.setUrl(url) 
            scene.addWidget(webview)
            map_view.setScene(scene)

            #disable the optimisation button
            optimise_route_button=self.route_optimisation_menu.itemAt(2).widget()
            optimise_route_button.setEnabled(False)

        elif self.route_entered==False:

            #display a message if no route was entered
            self.DisplayMessageBox("No route has been entered")
            
    #checks the route entered follows the correct format and that the 
    # waypoints entered are all present in the database
    def CheckOriginalRoute(self):

        #error message initialisation
        error_msg=""

        original_route_menu=self.route_optimisation_menu.itemAt(1)
        route_text_edit=original_route_menu.itemAt(0).widget()
        original_route=route_text_edit.toPlainText()

        if original_route=="":
            error_msg=error_msg+"No route has been entered \n"
            self.ready_to_optim=False
        else:
            pattern = r'^[A-Z0-9]+(-[A-Z0-9]+)*$'
            #check if the format is correct, needs to be ABCD-EFGH-IJKL
            if re.match(pattern, original_route):
                #split the route into a list of waypoints using - as seperator
                original_route=original_route.split("-")
                original_route_series = pd.Series(original_route)

                # Check if all elements of original_route are present in the DataFrame column 'Point'
                check = original_route_series.isin(self.clustered_waypoints_df['Point']).all()
                false_indices = np.where(~check)[0]

                if len(false_indices)>0:
                    error_msg=error_msg+f"{len(false_indices)} waypoint(s) of the waypoints entered are non existent"
                    self.ready_to_optim=False
                else:
                    #ready to optimise
                    self.ready_to_optim=True
                    original_route.insert(0,"take_off")
                    original_route.append("landing")
                    self.original_route=original_route
                    self.DisplayMessageBox("Route Successfully Entered")

            else:
                #if the format is not correct
                error_msg=error_msg+"The route entered does not follow the expected format"
                self.ready_to_optim=False

        if error_msg!="":
            self.DisplayMessageBox(error_msg)

    #checks that user didn't select same airport twice or forgot to select an airport
    def CheckFlightDetails(self):

        #same logic as in above function
        error_msg=""
        departure=self.flight_details_menu.itemAt(2).widget().currentText()
        arrival=self.flight_details_menu.itemAt(7).widget().currentText()

        if departure=="Select Airport":
            error_msg=error_msg+"No departure airport selected \n"
            self.ready_to_submit=False
        if arrival=="Select Airport":
            error_msg=error_msg+"No arrival airport selected \n"
            self.ready_to_submit=False
        if error_msg=="" and departure==arrival:
            error_msg=error_msg+"Departure and arrival airports cannot be the same \n"
            self.ready_to_submit=False
        
        if error_msg=="":
            self.DisplayMessageBox("Flight Details Accepted")
            self.ready_to_submit=True
        else:
            self.DisplayMessageBox(error_msg)
    
    #displays a message box given the message entered as a string
    def DisplayMessageBox(self,message):
        msg=QMessageBox()
        msg.setText(message)
        msg.setWindowTitle("Message")
        msg.exec_()

#function which generates a precipitation field based on a normal distribution
#inputs are mean values of lat and lon and number of points. 
def GenerateGaussianPrecipitation(lat,lon,npoints):
    
    #standard deviation for position and precipitation
    std_dev_pos=0.05
    std_dev_precip=0.1

    #generate data
    data = np.random.normal(size=(npoints, 3)) * np.array([[std_dev_pos, std_dev_pos, std_dev_precip]]) + np.array([[lat, lon, 1]])

    #absolute value of p so that all values are guaranteed positive
    data[:,2]=abs(data[:,2])

    return data

''' 
NOTe: the below precipitation clusters around glossop, birmingham and liverpool
can be modified and replaced by other clusters as well.
'''
#number of points in each cluster of precipitation.
n=100
precip_glossop=GenerateGaussianPrecipitation(53.444631, -1.955006,n)
precip_birmingham=GenerateGaussianPrecipitation(52.490128, -2.010586,n)
precip_liverpool=GenerateGaussianPrecipitation(53.431863, -2.983356,n)

#store in a single array
precip= np.vstack((precip_glossop, precip_birmingham, precip_liverpool))

def main():
    app= QApplication([])
    window=FlightPlanner(precipitation_matrix=precip)
    app.exec()

if __name__== '__main__':
    main()