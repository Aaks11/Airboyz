from faker import Faker
import random
import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import math
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

f = Faker()

# step1 is to determine the fields we want to have:
# 1. Weather ID (see on openweather for documentation) https://openweathermap.org/weather-conditions
# 2. Main Weather (Rain, Snow, Clouds etc)
# 3. Temperature (C)
# 4. Pressure (atm) -> don't consider this for now
# 5. Humidity (%)
# 6. Visibility (m)
# 7. Wind Speed (m.s-1) --> refers to an average (see gust below)
# 8. Wind Direction (deg)
# 9. Wind Gust (m.s-1)
# 10. Cloud Cover (%)
# 11. Timestamp

class FakeGridWeather:
    #self refers to all the data across the grid

    # Create base levels and bounds for randomness of each variable (if applicable)
    base={
        "Temperature": 10,
        "Humidity": 50,
        "Visibility": 2000,
        "Wind Speed": 10,
        "Wind Direction": 120,
        "Wind Gust" : 20,
        "Cloud Cover": 40
        }

    bounds={
        "Temperature": [-2,2],
        "Humidity": [-20,20],
        "Visibility": [-1000,1000],
        "Wind Speed": [-5,5],
        "Wind Direction": [-30,30],
        "Wind Gust":[-5,5],
        "Cloud Cover":[-20,20] 
        }

    #create possible weather states as lists (add more when we know what the possible states are as openweather doesn't provide a full list)
    states={
        "Main": ["Rain", "Snow", "Clouds"]
    }

    #save data in JSON file or not
    SAVE=False
    
    #constructor, ub is upper bound and lb is lower bound. n is number of points in each direction which is equal
    def __init__(self, n , lbX, ubX, lbY, ubY):

        #create vectors in each direction
        self.x= np.linspace(lbX, ubX, n)
        self.y=np.linspace(lbY, ubY, n)

        #create mesh objects
        self.X,self.Y=np.meshgrid(self.x,self.y)

        #total number of points on grid
        self.n=n
        self.npoints=n**2

        self.PlottingVar=[]
        self.FakeWeatherMap=[]
    
    #function to create fake data, takes in the base levels bounds and discrete states. 
    def FakeDataCreator(self):

        #quantities = base_quantity + random within bounds
        temperature = self.base["Temperature"]   +random.uniform(self.bounds["Temperature"][0],self.bounds["Temperature"][1])
        humidity = self.base["Humidity"]         +random.uniform(self.bounds["Humidity"][0],self.bounds["Humidity"][1])
        visibility = self.base["Visibility"]     +random.uniform(self.bounds["Visibility"][0],self.bounds["Visibility"][1])
        wind_speed = self.base["Wind Speed"]     +random.uniform(self.bounds["Wind Speed"][0],self.bounds["Wind Speed"][1])
        wind_dir = self.base["Wind Direction"]   +random.uniform(self.bounds["Wind Direction"][0],self.bounds["Wind Direction"][1])
        wind_gust = self.base["Wind Gust"]       +random.uniform(self.bounds["Wind Gust"][0],self.bounds["Wind Gust"][1])
        cloud_cover = self.base["Cloud Cover"]   +random.uniform(self.bounds["Cloud Cover"][0],self.bounds["Cloud Cover"][1])

        #discrete states
        main = random.choice(self.states["Main"])

        #create dictionary with all fake data, mocking an API response
        FakeData={
            "Main": main,
            "Temperature": temperature,
            "Humidity": humidity,
            "Visibility" : visibility,
            "Wind Speed": wind_speed,
            "Wind Direction": wind_dir,
            "Wind Gust": wind_gust,
            "Cloud Cover": cloud_cover
        }
        if (self.SAVE == True):# save file as weather.json in local directory
            with open("weather.json", 'w') as f:
                json.dump(FakeData, f, indent=2)
        
        # return fake data as dictionary
        return FakeData

    # method that generates the fake data for n number of points    
    def FakeWeatherDomain(self):

        for _ in range(0,self.npoints):
            self.FakeWeatherMap.append(self.FakeDataCreator())
    
    #method that stores one of the weather variables in a list to be plotted, only to be used after FakeWeatherDomain()
    def StoreVariable(self, variable_str):
        if (variable_str in self.base.keys()):
            for i in range(0,self.npoints):
                self.PlottingVar.append(self.FakeWeatherMap[i][variable_str])
            self.PlottingVar=np.array(self.PlottingVar)
            self.PlottingVar=np.reshape(self.PlottingVar,(self.n,self.n))
        else: 
            raise ValueError("Weather variable not found!")
    
    #method that plots the wind direction
    def PlotWindDir(self):
        
        self.StoreVariable("Wind Direction")
        self.PlottingVar=self.PlottingVar.flatten()
        #creates U an V vectors
        #convert angles to radians
        angles=(360- self.PlottingVar+90)*math.pi/180
        dir=np.zeros((self.npoints,2))
        for i in range(0,self.npoints):
            dir[i,:]=[math.cos(angles[i]),math.sin(angles[i])]
        U=np.reshape(dir[:,0],((self.n),self.n))
        V=np.reshape(dir[:,1],((self.n),self.n))
        return U,V    

xmin, xmax = -1 , 1
ymin, ymax = -2, 1

test=FakeGridWeather(20,xmin,xmax,ymin,ymax)
test.FakeWeatherDomain()
# U,V=test.PlotWindDir()

# img= plt.imread("radar.jpg")
# fig, ax = plt.subplots()
# ax.set_facecolor('black')
# plt.quiver(test.X, test.Y,U,V,color="white",zorder=2)
# imagebox = OffsetImage(img, zoom=0.5)
# ab = AnnotationBbox(imagebox, (0.5, 0.5), xycoords='axes fraction',bboxprops=dict(facecolor='black', edgecolor='black'))
# ax.add_artist(ab)

# plt.savefig('output_plot.png', transparent=False)
# plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
#                 labelbottom=False, labeltop=False, labelleft=False, labelright=False)
# plt.gca().spines['bottom'].set_visible(False)

# plt.show()

test.StoreVariable("Temperature")

plt.contourf(test.X, test.Y, test.PlottingVar)
plt.show()
plt.colorbar()