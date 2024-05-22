# %%
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
from pde import solve_laplace_equation, PolarSymGrid
from pde import CartesianGrid as XYGrid
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# +---------------------------+
# |Created for project Airboyz|
# +---------------------------+

# This code is intended to create fake data for testing purposes and later be modified to 
# include API data. It takes the form of a class and considers the following weather properties:

# 1. Temperature (C) --> it is obtained by solving the Laplace Equation on a cartesian grid around the aircraft
# 2. Pressure (atm) -> assumed constant at a certain altitude level and used to calculate pressure altitude
# 3. Density --> obtained using the ideal gas law from Temperature
# 4. Humidity (%) --> 
# 5. Visibility (m) --> obtained the same way as temperature for a smooth distribution
# 6. Wind Speed (m.s-1) --> refers to an average (see gust below). not sure how obtain yet
# 7. Wind Direction (deg) --> random function can be used to generate
# 8. Wind Gust (m.s-1) --> same as wind speed
# 9. Cloud Cover (%) --> can be related to visibility maybe? or opposite way?

class DataCreator:

    R_constant=(287,)
     
    def __init__(self,radius,radius_grid,n,n_grid,altitude_pressure):
        if radius<=0:
            raise ValueError("radius around aircraft cannot be <=0")
        else:
            self.radius=radius
        
        if radius_grid<self.radius:
            raise ValueError("solution grid limits cannot be less than aircraft domain")
        else:
            self.radius_grid=radius_grid

        if isinstance(n, int)==False or n==0:
            raise ValueError("n cannot be a float or 0")
        else: 
            self.n=n
        
        if isinstance(n_grid, int)==False or n_grid==0:
            raise ValueError("grid number of points cannot be a float or 0")
        else:
            self.n_grid=n_grid
         
        if altitude_pressure<=0:
            raise ValueError("altitude pressure cannot be negative or 0")
        else:
            self.altitude_pressure=(altitude_pressure,)
        
        #mesh in each direction
        self.rmesh=np.linspace(0,self.radius,self.n)
        self.theta_mesh=np.linspace(0,2*np.pi,self.n)
        #overall mesh
        self.rmesh,self.theta_mesh=np.meshgrid(self.rmesh,self.theta_mesh)
        #X and Y of interest (used later for interpolation)
        self.X=self.rmesh*np.cos(self.theta_mesh)
        self.Y=self.rmesh*np.sin(self.theta_mesh)
        #makes it into a list of x-y points
        self.positions=np.transpose(np.vstack([self.X.ravel(), self.Y.ravel()]))

        self.CartesianGrid=XYGrid([[-self.radius_grid, self.radius_grid]] * 2,self.n_grid)

    # method can be used after initialisation
    def GetFakeTemperature(self,bcs_x,bcs_y):
        
        #format of bcs_x and bcs_y
        #bc_x = [{"value": 10}, {"value": 11}] where first value is lower x and second is upper x
        #bc_y = [{"value": 12}, {"value": 13}] same with y
        #boundary conditions
        self.T_bc_x = bcs_x
        self.T_bc_y = bcs_y
        self.T_bcs= [self.T_bc_x, self.T_bc_y]

        #solution
        self.T_grid=solve_laplace_equation(self.CartesianGrid, self.T_bcs)

        #interpolate solution to polar domain
        self.T_fake=np.reshape(self.T_grid.interpolate(self.positions),(self.n,self.n))

    def IdealGasLaw(self,T):
        return self.altitude_pressure[0]/(self.R_constant*(T+273.15))

    #get density from temperature
    def GetFakeDensity(self):
        self.FakeDensity=self.IdealGasLaw(self.T_fake)
    
    #plot contours of temperature
    def PlotTemperature(self,mode):
        fig, ax = plt.subplots(figsize=(8,6),subplot_kw={'projection': 'polar'})
        ax.xaxis.grid(True)
        ax.xaxis.set_ticklabels([])
        ax.set_rmax(self.radius)
        ax.set_rticks(np.array([0.25,0.5,0.75,1])*self.radius)
        #mode refers to fake or real mode,
        if mode=="fake":
            contours=ax.contourf(self.theta_mesh,self.rmesh,self.T_fake,alpha=1,zorder=1,cmap='hot')
        cbar = fig.colorbar(contours, ax=ax)
        plt.show()
    
    #plot contours of density
    def PlotDensity(self,mode):
        fig, ax = plt.subplots(figsize=(8,6),subplot_kw={'projection': 'polar'})
        ax.xaxis.grid(True)
        ax.xaxis.set_ticklabels([])
        ax.set_rmax(self.radius)
        ax.set_rticks(np.array([0.25,0.5,0.75,1])*self.radius)
        #mode refers to fake or real mode,
        if mode=="fake":
            contours=ax.contourf(self.theta_mesh,self.rmesh,self.FakeDensity,alpha=1,zorder=1,cmap='hot')

        cbar = fig.colorbar(contours, ax=ax)
        plt.show()
    
    # get fake wind
    def GetFakeWind(self,base_velocity,lb_velocity,ub_velocity,base_direction,lb_direction,ub_direction):
        self.FakeWindDirection=base_direction + np.random.uniform(lb_direction,ub_direction,(self.n,self.n))
        self.FakeWindDirection=(360-self.FakeWindDirection+90)*np.pi/180
        self.FakeWindVelocities=base_velocity + np.random.uniform(ub_velocity,ub_velocity,(self.n,self.n))


    def PlotWind(self,mode):
        # fig, ax = plt.subplots(figsize=(8,6),subplot_kw={'projection': 'polar'})
        fig, ax = plt.subplots(figsize=(8,6))
        ax.xaxis.grid(True)
        ax.xaxis.set_ticklabels([])
        ax.axis('off')
        # ax.set_rmax(self.radius)
        # ax.set_rticks(np.array([0.25,0.5,0.75,1])*self.radius)

        if mode=="fake":
            x_velocities=self.FakeWindVelocities*np.cos(self.FakeWindDirection)
            y_velocities=self.FakeWindVelocities*np.sin(self.FakeWindDirection)

            r_velocities=x_velocities*np.cos(self.FakeWindDirection)+y_velocities*np.sin(self.FakeWindDirection)    
            theta_velocities=-x_velocities*np.sin(self.FakeWindDirection)+y_velocities*np.cos(self.FakeWindDirection) 
        X,Y=np.meshgrid(np.linspace(-self.radius,self.radius,self.n),np.linspace(-self.radius,self.radius,self.n))
        lw=self.FakeWindVelocities/np.max(self.FakeWindVelocities)
        # stream=ax.streamplot(self.rmesh, self.theta_mesh, x_velocities, y_velocities, color=y_velocities,density=[1],cmap='autumn',linewidth=lw,broken_streamlines=True) 
        # stream=ax.streamplot(np.transpose(self.theta_mesh), np.transpose(self.rmesh), theta_velocities, r_velocities, color=self.FakeWindVelocities,density=[1],cmap='autumn',linewidth=lw,broken_streamlines=True)
        stream=ax.streamplot(X, Y, x_velocities, y_velocities, color=self.FakeWindDirection,density=[1],cmap='autumn',linewidth=lw,broken_streamlines=True)
        ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
        c = plt.Circle((0, 0), radius=self.radius, facecolor='black') 
        ax.add_patch(c)
        # ax.figure.subplots_adjust(bottom=-1, top=1, left=-1, right=1)
        plt.show() 


# %%
test=DataCreator(1,1,40,100,101325)
bc_x = [{"value": 10}, {"value": 11}]
bc_y = [{"value": 12}, {"value": 13}]
test.GetFakeTemperature(bc_x,bc_y)
test.GetFakeDensity()
test.GetFakeWind(10,-5,5,120,-50,50)
theta=test.theta_mesh
# test.PlotWind("fake")
test.PlotDensity("fake")
test.PlotTemperature("fake")
#%%



        