# this file is used to define the class and later imported into the "main_file.py"
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

class DataHandler:

    def __init__(self,radius,**kwargs):
        
        self.radius=radius
        self.flight_number=kwargs.get('flight_number', None)
        self.test_mode=kwargs.get('test_mode',None)

        if self.flight_number and self.test_mode is None:
            raise TypeError("Argument missing, either flight_number must be defined the instance used in testing mode")


    # def GetFlightInfo(self):