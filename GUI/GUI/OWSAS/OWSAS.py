from PyQt5.QtWidgets import *
from PyQt5.QtGui import QFont, QPainter, QBrush, QColor, QPixmap, QScreen
from PyQt5 import uic
from PyQt5.QtCore import QTimer, QTime, QDateTime, QUrl, Qt, QRectF, QSize
from PyQt5.QtWebEngineWidgets import QWebEngineView
import folium
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pandas as pd
from class_definition import CentralDataHandler, FlightData, FlightPlanning, NearbyMETARInfo
import re
from scipy.interpolate import interp1d
from PIL import Image
import warnings
from folium import raster_layers
warnings.filterwarnings("ignore")
import requests
import json
from metar import Metar
import numpy as np
import os

'''
   +----------------------------------------------------------------+
   | Code Structure, UI Design and Class Definition by Jade Nassif  |
   | Warning System by Aakalpa Deepak Savant                        |
   | Precipitation Visualisation by Amey Shah                       |
   |                                                                |
   | Aerospace Computational Engineering MSc 2023-2024              |
   | Cranfield University                                           |
   | jade.nassif.385@cranfield.ac.uk                                |
   | aakalpadeepak.savant.657@cranfield.ac.uk                       |
   | amey.shah.372@cranfield.ac.uk                                  |
   +----------------------------------------------------------------+ 
'''

''' 
This code makes use of the four classes defined in the class_definition file to 
define the OWSAS GUI and its functions. The UI can be launched by running this file
(assuming all necessary files are in your local directory)

The OWSAS GUI is a QMainWindow that is built using the UI file created in Qt Designer
It can take either of the following as inputs:
1. Flight number, aircraft type and airline ICAO (following the FlightData Class format)
2. A predefined trail .csv which contains the flight information at every timestamp of the flight

It then requests wind data, METAR data and precipitation contours heatmaps and displays this in a 
user friendly manner 
'''

class OWSAS_GUI(QMainWindow):

    # constructor
    def __init__(self):

        super(OWSAS_GUI,self).__init__()

        #load the UI file
        uic.loadUi("OWSAS.ui",self)
        
        self.show()
        self.setWindowTitle("OWSAS GUI")

        #the function below rounds the corners and set the style of the UI
        self.RoundButtons()

        #initialise default plotting mode and view
        self.view="true north"
        self.plot_mode="gradient"

        #set the buttons defining the plotting mode and view to their default values
        self.gradient_based_button.setChecked(True)
        self.current_view.setText(f"Current View : North Centered")

        #link gradient and speed based buttons to their respective functions defined below
        self.speed_based_button.toggled.connect(self.SpeedBasedButton)
        self.gradient_based_button.toggled.connect(self.GradientBasedButton)

        #link action of loading in a trail file to the corresponding function
        self.actionTrail.triggered.connect(self.OpenTrail)

        #link action of submitting flight details to corresponding function
        self.submit_flight_details_button.clicked.connect(self.SubmitFlightDetails)

        #toggle button action
        self.toggle_view_button.clicked.connect(self.ToggleButton)

        #link start/refresh button to function
        self.refresh_button.clicked.connect(self.StartRefresh)

        # timer that starts the clock 
        self.elapsed_time = QTime(0, 0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.UpdateTime)
        self.timer.start(1000)  # Update every one second
        
        # Initial update of time label
        self.UpdateTime()
    
    #function which defines what happens when the open --> trail action is selected
    def OpenTrail(self):
        
        #opens a dialog box and gets the file path
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File", "", "Comma Seperated Files (*.csv)")

        #check if a file was selected
        if file_path!="":

            trail=pd.read_csv(file_path)
            headers=['lat','lon','alt','timestamp','heading','ground speed']

            #check that the loaded file matches the expected format
            if list(trail.columns)==headers:
                
                #store the trail as an attribute
                self.trail=trail

                #set the variable which checks whether or not the trail is realtime to false
                self.realtime_trail=False

                #display following message
                message="Trail successfully loaded"
                self.DisplayMessageBox(message)

                #disable the submit flight details workflow and loading trail
                self.flight_no_edit.setEnabled(False)
                self.aircraft_icao_edit.setEnabled(False)
                self.airline_icao_edit.setEnabled(False)
                self.submit_flight_details_button.setEnabled(False)
                self.actionTrail.setEnabled(False)

                #obtain the base timestamp at which time starts
                self.base_timestamp= self.trail['timestamp'].iloc[-1]
            else:
                message="Data incorrectly formatted! Please try again"
                self.DisplayMessageBox(message)
            
    #function which defines what happens after flight details are submitted into the UI
    def SubmitFlightDetails(self):

        #set the default error message to ""
        error_msg=""

        #check that the flight number entered matches the expected pattern
        pattern_flightno = r'^[A-Za-z]{1,2}\d{1,4}$'
        if re.match(pattern_flightno,self.flight_no_edit.text()):
            self.flight_no_edit.setEnabled(False)
        elif self.flight_no_edit.text()=="":
            error_msg="Flight number is missing\n"
        else:
            error_msg="Incorrect format for flight number\n"
        
        #same for the aircraft icao
        pattern_aircraftICAO= r'^[A-Z]{1}[A-Z0-9]{3,4}$'
        if re.match(pattern_aircraftICAO,self.aircraft_icao_edit.text()):
            self.aircraft_icao_edit.setEnabled(False)

        elif self.aircraft_icao_edit.text()=="":
            error_msg=error_msg+"Aircraft ICAO is missing\n"
        else:
            error_msg=error_msg+"Incorrect format for aircraft type\n"

        #same for airline icao
        pattern_airlineICAO=r'^[A-Z]{3}$'
        if re.match(pattern_airlineICAO,self.airline_icao_edit.text()):
            self.airline_icao_edit.setEnabled(False)
        elif self.airline_icao_edit.text()=="":
            error_msg=error_msg+"Airline ICAO is missing"
        else:
            error_msg=error_msg+"Incorrect format for airline ICAO"
        
        #if no errors are present, disable the workflow of the submit flight details and loading a trail
        if error_msg=="":

            self.submit_flight_details_button.setEnabled(False)
            self.actionTrail.setEnabled(False)
            #set the realtime trail to True
            self.realtime_trail=True

        else:
            self.DisplayMessageBox(error_msg)
        

    #function to check if radius value is valid and return status        
    def CheckRadiusValue(self):

        #checks if the radius has not been checked yet
        if self.radius_edit.isEnabled():
            #check if the input is a number in the try statement
            try:
                radius=float(self.radius_edit.text())
                #50km lower limit 
                if radius<50:

                    message="Please enter a positive radius greater than 50 km"
                    self.DisplayMessageBox(message)

                    radiusCheck=False
                else:
                    self.radius_edit.setEnabled(False)
                    radiusCheck=True
            #should it not be a float, return an error
            except:
                radiusCheck=False
                if self.radius_edit.text()=="":

                    message="No radius has been entered"
                    self.DisplayMessageBox(message)
                    
                else:
                    message="Radius entered is not a number"
                    self.DisplayMessageBox(message)
                    
        else:
            #if box is already disabled then the radius has been previously checked
            radiusCheck=True
        return radiusCheck

    #most important function which defines what happens when the start/referesh button is clicked
    def StartRefresh(self):
        
        #check radius has been entered (see above definition)
        radiusCheck=self.CheckRadiusValue()

        if radiusCheck==True and hasattr(self,"realtime_trail"):

            #define the width of the windplot. Assuming a default DPI of 100
            width=self.wind_plot.width()/100
            height=self.wind_plot.height()/100

            #comb1=true north+gradient
            #comb2=true north+speed
            #comb3=heading+gradient
            #comb4=heading+speed

            #if flight details have been entered
            if self.realtime_trail==True:
                
                # the line below returns an instance of the FlightData class which contains the current flight information
                flight=self.GetRealtimeFlightInfo()

                #save the current trail as a csv file, import it as a dataframe in self.trail and then delete again
                #until next time the user clicks refresh
                if os.path.exists(f"flight_{self.flight_no_edit.text()}.csv"):
                    os.remove(f"flight_{self.flight_no_edit.text()}.csv")

                flight.TrailtoCSV()
                
                self.trail=flight.TrailtoDf(f"flight_{self.flight_no_edit.text()}.csv")
                os.remove(f"flight_{self.flight_no_edit.text()}.csv")

                
                #extract the current info dictionary
                current_info=flight.current_info
                print("Flight information obtained")
            # if a trail has been opened
            elif self.realtime_trail==False:
                
                #the line below returns an instance of the FlightPlanning class which contains the trail file
                flight=self.GetTrailInfo()
                #interpolate to the current flight conditions x seconds after the base timestamp
                current_info=self.InterpolateTrailInfo()
                print("Flight information interpolated")

            #extract the current info
            lat=current_info["lat"]
            lon=current_info["lon"]
            heading=current_info["heading"]
            ground_speed=current_info["ground speed"]
            altitude=current_info["alt"]
            timestamp=current_info["timestamp"]
            
            #display the current information on the dashboard
            self.DisplayInfo(ground_speed,altitude,heading,lat,lon)
            print("Dashboard information displayed")

            #get the METAR information
            raw_metar_text,metar_coordinates,wind_direction,wind_speed=self.GetMETARInfo(lat,lon)

            #generate the warnings and display them, threshold value of 35 knots
            self.GenerateWarnings(wind_direction,wind_speed,heading,35)
            self.UpdateWarnings()
            print("Warnings Updated")

            #update the map to the current position and zoom, by default zoom is 7 but can be adjusted
            zoom=7
            self.UpdateMap(lat,lon,zoom,heading,timestamp,raw_metar_text,metar_coordinates)
            print("Map Updated")

            #modify the flight object into two versions, one for the true north mesh and one of the heading mesh
            #current info is also input here for the trail file as it requires lat,lon
            # '''
            flight_true_north=self.GetRealtimeWindData(flight,"true north",current_info)
            flight_heading=self.GetRealtimeWindData(flight,"heading",current_info)
            print("Wind data obtained")

            #obtain the four possible figures corresponding to the combinations described 
            # at the begininning of this if clause
            self.WindPlot_1=self.GetWindPlot(flight_true_north,"gradient",width,height,"true north")
            self.WindPlot_2=self.GetWindPlot(flight_true_north,"speed",width,height,"true north")
            self.WindPlot_3=self.GetWindPlot(flight_heading,"gradient",width,height,"heading")
            self.WindPlot_4=self.GetWindPlot(flight_heading,"speed",width,height,"heading")

            #set the wind check to True as it is required in the PlotWind function 
            self.wind_check=True
            
            #plot the wind based on the current values of self.view and self.plot_mode
            self.PlotWind()
            # '''
            print("Wind Plotted")
        
        #case where neither flight details nor a trail have been entered
        elif radiusCheck==True and hasattr(self,"realtime_trail")==False :
            self.wind_check=False
            message="Input is missing or incorrect"
            self.DisplayMessageBox(message)

    # function which obtains the current flight information
    # returns a FlightData object
    def GetRealtimeFlightInfo(self):
        
        #get inputs
        radius=float(self.radius_edit.text())
        flight_number=self.flight_no_edit.text()
        aircraft_type=self.aircraft_icao_edit.text()
        airline_ICAO=self.airline_icao_edit.text()
        flight=FlightData(radius,flight_number,aircraft_type,airline_ICAO)

        #get the current flight information
        flight.GetCurrentFlightInfo()
        return flight
    
    #function which obtains the trail information and stores it in a FlightPlanning object
    def GetTrailInfo(self):
        radius=float(self.radius_edit.text())

        #initialise the flight planning instance
        flight=FlightPlanning(radius,self.trail)
        return flight

    # function which obtains the wind data
    # takes in either a FlightData object or a FlightPlanning object as both inherit common functions from
    # CentralDataHandler
    def GetRealtimeWindData(self,flight,view_mode,current_info):
        
        #default number of points in each direction for the mesh
        n=10

        radius=float(self.radius_edit.text())

        if self.realtime_trail==True:

            flight.GenerateMesh(n,view_mode)
        
        elif self.realtime_trail==False:

            lat=current_info["lat"]
            lon=current_info["lon"]
            heading=int(current_info["heading"])
            
            flight.GenerateMesh(lat,lon,heading,n,view_mode)

        #obtain the wind data for the mesh stored in the flight instance
        flight.GetWindData()

        return flight

    #function which defines what happens when the gradient-based radio button is selected
    def GradientBasedButton(self):
        #change the plot_mode attribute to gradient
        self.plot_mode="gradient"

        #if wind plots have been generated already, plot the wind
        if hasattr(self,"wind_check"):
            if self.wind_check==True:
                self.PlotWind()

    #similar to the above but for speed based
    def SpeedBasedButton(self):
        self.plot_mode="speed"
        if hasattr(self,"wind_check"):
            if self.wind_check==True:
                self.PlotWind()

    #define the style of the window    
    def RoundButtons(self):
        style_sheet = """

    QPushButton {
        background-color: white;
        border-radius: 10px;
        border: 1px solid black;
        font-size: 17px;
    }

    QLineEdit {
        border: 1px solid black;
        border-radius: 10px;
        background-color: white;
    }

    QFrame {
        border-radius: 10px;
        border: 1px solid black;
        background-color: white
    }
"""

        pixmap = QPixmap("background.jpg")
        # label to hold the background image
        self.background_label.setPixmap(pixmap)
        self.background_label.lower()

        # set the style sheet for the widgets
        self.submit_flight_details_button.setStyleSheet(style_sheet)
        self.refresh_button.setStyleSheet(style_sheet)
        self.toggle_view_button.setStyleSheet(style_sheet)
        self.flight_no_edit.setStyleSheet(style_sheet)
        self.aircraft_icao_edit.setStyleSheet(style_sheet)
        self.airline_icao_edit.setStyleSheet(style_sheet)
        self.radius_edit.setStyleSheet(style_sheet)
        self.warnings_frame.setStyleSheet(style_sheet)


    #defines what happens when the Toggle View button is clicked
    def ToggleButton(self):
        view=self.view
        if view=="heading":

            self.view="true north"
            self.current_view.setText(f"Current View : North Centered")

        elif view=="true north":

            self.view="heading"
            self.current_view.setText(f"Current View : Heading Centered")

        #if the wind plots have been generated, plot the wind
        if hasattr(self,"wind_check"):
            if self.wind_check==True:
                self.PlotWind()
    
    # function to plot the wind based on values of self.plot_mode and self.view 
    def PlotWind(self):

        if self.view=="true north" and self.plot_mode=="gradient":
            WindPlot=self.WindPlot_1
        elif self.view=="true north" and self.plot_mode=="speed":
            WindPlot=self.WindPlot_2
        elif self.view=="heading" and self.plot_mode=="gradient":
            WindPlot=self.WindPlot_3
        elif self.view=="heading" and self.plot_mode=="speed":
            WindPlot=self.WindPlot_4
        
        WindPlot.subplots_adjust(left=0, right=1, bottom=0, top=1)
        canvas = FigureCanvas(WindPlot)
        scene = QGraphicsScene()
        scene.addWidget(canvas)  # Add the canvas directly to the scene
        self.wind_plot.setScene(scene)
        self.wind_plot.setStyleSheet("border-radius: 30px;")

    #generate the wind plot corresponding to the selected parameters
    #takes in a FlightData or FlightPlanning object
    def GetWindPlot(self,flight,plot_mode,width,height,view):

        WindPlot=flight.PlotWind(plot_mode,width,height,view)
        return WindPlot

    #function which updates the time
    def UpdateTime(self):

        current_datetime = QDateTime.currentDateTimeUtc()
        # Format the current time as a string
        time_str = current_datetime.toString("hh:mm:ss")
        date_str= current_datetime.toString("yyyy-MM-dd")
        self.elapsed_time = self.elapsed_time.addSecs(1)
        # Update the time and date label text
        self.time_label.setText(time_str)
        self.date_label.setText(date_str)

    #function which interpolates the current flight info from the trail file
    def InterpolateTrailInfo(self):
        #check if the interpolation functions have already been created
        if hasattr(self,"interp_lat")==False:

            self.interp_lat=interp1d(self.trail['timestamp'],self.trail['lat'])
            self.interp_lon=interp1d(self.trail['timestamp'],self.trail['lon'])
            self.interp_alt=interp1d(self.trail['timestamp'],self.trail['alt'])
            self.interp_heading=interp1d(self.trail['timestamp'],self.trail['heading'])
            self.interp_ground_speed=interp1d(self.trail['timestamp'],self.trail['ground speed'])

        #obtain the seconds elapsed
        seconds=int(self.elapsed_time.toString("ss"))
        
        #interpolate at the current timestamp x seconds after which the base timestamp was created
        lat=self.interp_lat(int(self.base_timestamp)+seconds)
        lon=self.interp_lon(int(self.base_timestamp)+seconds)
        alt=self.interp_alt(int(self.base_timestamp)+seconds)
        heading=self.interp_heading(int(self.base_timestamp)+seconds)
        ground_speed=self.interp_ground_speed(int(self.base_timestamp)+seconds)

        #store the info in a dictionary
        current_info={
            "lat": lat,
            "lon":lon,
            "alt":alt,
            "heading":heading,
            "ground speed":ground_speed,
            "timestamp": int(self.base_timestamp)+seconds
        }
        return current_info
    
    # function which displays the current flight info on the dashboard
    def DisplayInfo(self,ground_speed,altitude,heading,latitude,longitude):

        self.ground_speed.setText(f"{int(ground_speed)}")
        self.altitude.setText(f"{int(altitude)}")
        self.heading.setText(f"{int(heading)}")
        self.latitude.setText(f"{latitude: .3f}")
        self.longitude.setText(f"{longitude: .3f}")

    #function which updates the map
    def UpdateMap(self,lat,lon,zoom,heading,timestamp,raw_metar_text,metar_coordinates):

        scene = QGraphicsScene()
        webview = QWebEngineView()
        webview.setFixedSize(self.precip_plot.width(), self.precip_plot.height())  # Set size to match map widget size


        #initialise a folium map
        m=folium.Map(location=[lat,lon],zoom_start=int(zoom),control_scale=True,tiles="cartodb dark_matter")

        #add circle of radius r
        radius=float(self.radius_edit.text())
        folium.Circle(location=[lat,lon],radius=radius*1e3).add_to(m)


        #add the METAR reports to the map as markers where the popups are the METAR reports
        if len(raw_metar_text)>=1:
            for i in range(len(raw_metar_text)):

                popup_text = f"<div style='font-size: 16px;'>{raw_metar_text[i]}</div>"
                # folium.Marker(location=[metar_coordinates[i][1], metar_coordinates[i][0]], popup=raw_metar_text[i],icon=folium.Icon(color='red',icon_size=(50,50))).add_to(m)
                folium.Marker(location=[metar_coordinates[i][1], metar_coordinates[i][0]],icon=folium.CustomIcon(icon_image="airport_blue.png", icon_size=(30, 30)),popup=popup_text).add_to(m)
        # get the trail so far as a dataframe
        trail_so_far=self.trail[self.trail['timestamp'] < timestamp]
        trail_so_far=trail_so_far[["lat","lon","timestamp"]]

        current_info={
            "lat":lat,
            "lon": lon,
            "timestamp": timestamp
            }

        #add the current info the dataframe
        new_row = pd.DataFrame(current_info, index=[0])
        trail_so_far=pd.concat([new_row,trail_so_far]).reset_index(drop=True)

        #add the trail so far to the map as a polyline
        points = trail_so_far[["lat","lon"]].values
        folium.PolyLine(locations=points, color='blue').add_to(m)
        
        # obtain precipitation layers and return the updated map
        frames, data = self.GetWeatherFrames(kind='radar')
        if frames:
            latest_frame = frames[0]
            weather_layer,m = self.CreateLayer(latest_frame, data, kind='radar', base_map=m)

        #open the png image corresponding to the aircraft icon
        image = Image.open("aircraft_grey.png")
        #rotate the image to the heading of the aircraft
        rotated_aircraft = image.rotate(-heading, expand=True)
        rotated_aircraft.save("rotated_aircraft.png")
        #add the aircraft icon to the map
        folium.Marker(location=[lat,lon],icon=folium.CustomIcon(icon_image="rotated_aircraft.png", icon_size=(100, 100))).add_to(m)
        #save the map
        m.save('UI_map.html')
        #display the map
        current_directory = os.getcwd()
        file_name = "UI_map.html"
        full_path = os.path.join(current_directory, file_name)
        url=QUrl.fromLocalFile(full_path)
        webview.setUrl(url) 
        scene.addWidget(webview)
        self.precip_plot.setScene(scene)
        self.precip_plot.setStyleSheet("border-radius: 30px;")
    
    #function which displays an arbitary message box
    def DisplayMessageBox(self,message):
        msg=QMessageBox()
        msg.setText(message)
        msg.setWindowTitle("Message")
        msg.exec_()
    
    #function which gets the metar info for a given radius
    #uses the NearbyMETARInfo class to get the metar info
    #return list of raw metar reports, list coordinates of METAR station, lists of wind directions and wind speed
    def GetMETARInfo(self,lat,lon):
        radius=float(self.radius_edit.text())
        metar=NearbyMETARInfo(radius,lat,lon)
        metar.GetNearbyMETARInfo()
        raw_metar_text=metar.GetRawMETARList()
        metar_coordinates=metar.GetMETARCoordinates()
        wind_direction,wind_speed=metar.GetWindProperties()
        
        return raw_metar_text,metar_coordinates,wind_direction,wind_speed
    
    #function which generates warnings. takes in list of wind directions and list of wind speed,
    # the heading of the aircraft and a threshold value of wind speed in kts above which to warn
    def GenerateWarnings(self,wind_direction,wind_speed,heading,threshold):
        
        #initialise totals for each
        #left crosswind, right crosswind and headwind
        left_cross=0
        right_cross=0
        headwind=0

        #iterate through the list
        for i in range(len(wind_direction)):

            # check no None values
            if wind_direction[i] is not None and wind_speed[i] is not None:

                #determine which type of wind is dominant component
                warn=self.WarningCheck(wind_direction[i],wind_speed[i],heading,threshold)
                
                #update totals
                if warn=="left cross":
                    left_cross+=1
                elif warn=="right cross":
                    right_cross+=1
                elif warn=="headwind":
                    headwind+=1
        
        #store totals in a dictionary
        self.warnings_all={
            "Left Crosswind":left_cross,
            "Right Crosswind":right_cross,
            "Headwind":headwind
        }
    
    #function which updates the warning box
    def UpdateWarnings(self):

        #initialise warning message
        message=""

        #iterate through te keys (wind types of the dictionary generated in the above function)
        for key,value in self.warnings_all.items():

            #if at least one instance of this type exists, add it to the warning message
            if value>0:
                message+=f"Strong {key} reported at {value} locations\n"

        #set the text in the warning box
        self.warnings_box.setText(message)

    # function which calculates the new bearing value obtained from heading+angle. 
    # used because sometimes the bearing obtained is negative or greater than 360
    def CalculateBearing(self,heading,angle):

        new_bearing=heading+angle #angle needs to be negative if going anticlockwise
        if new_bearing <0 :
            new_bearing=new_bearing+360
        return new_bearing

    '''
    function which determines which component of wind is dominant and if to warn about it
    inputs:
    - wind_direction in degrees
    - wind speed in knots
    - heading in degrees
    - threshold value for wind speed in knots
    '''
    def WarningCheck(self,wind_direction,wind_speed,heading,threshold):
        
        #determine the wind direction in the local x-y axes of the aircraft
        alpha=self.CalculateBearing(heading+90,-wind_direction)*np.pi/180

        #determine cross wind (+ve x) and tailwind (+ve y) components
        cross=wind_speed*np.cos(alpha)
        tailwind=wind_speed*np.sin(alpha)

        #check which is the dominant between the two
        if abs(cross)>abs(tailwind):
            
            #speed of dominant component
            dom_speed=abs(cross)

            # +ve component means wind hits from left of aircraft
            # -ve means hits from right of aircraft
            if cross>0:
                dominant="left cross"
            else:
                dominant="right cross"
        else:

            #speed of dominant component
            dom_speed=abs(tailwind)

            # +ve means tailwind 
            # -ve means headwind
            if tailwind>0:
                dominant="tailwind"
            else:
                dominant="headwind"
        
        # if the dominant component is not tailwind and is above the threshold,
        # register a warning in the dominant direction. otherwise return False
        if dominant!="tailwind" and dom_speed>threshold:
            return dominant
        else:
            return False                

    # function which creates the layer for the precipitation image to be displayed on the map
    def CreateLayer(self,frame, data, kind='radar', base_map=None, tile_size=512, color_scheme=6, smooth_data=1, show_snow=1):

        #setting up the rainview API with basic parameters needed
        if kind == 'satellite':
            color_scheme = 0
            smooth_data = 0
            show_snow = 0
        url = f"{data['host']}{frame['path']}/{tile_size}/{{z}}/{{x}}/{{y}}/{color_scheme}/{smooth_data}_{show_snow}.png"
        layer = raster_layers.TileLayer(
            tiles=url, name=f"{frame['time']}", attr='RainViewer', overlay=True, control=True, opacity=0.3  #to see the map below you can use opacity=0.5
        )
        if base_map:
            layer.add_to(base_map)
        return layer, base_map
    
    # function which obtains the weather frames from the Rainviewer API to be used to create the layers
    # in the above function
    def GetWeatherFrames(self,kind='radar'):
        
        #setting up the radar images and creating API calls
        url = 'https://api.rainviewer.com/public/weather-maps.json'
        response = requests.get(url)
        data = json.loads(response.text)
        if kind == 'satellite' and 'satellite' in data and 'infrared' in data['satellite']:
            frames = data['satellite']['infrared'] #creating JSON frame to impose on the map. 
        elif 'radar' in data and 'past' in data['radar']:
            frames = data['radar']['past']
            if 'nowcast' in data['radar']:
                frames.extend(data['radar']['nowcast'])
        else:
            return None, None
        
        return frames, data
    
def main():
    app= QApplication([])
    window=OWSAS_GUI()
    app.exec()

if __name__== '__main__':
    main()