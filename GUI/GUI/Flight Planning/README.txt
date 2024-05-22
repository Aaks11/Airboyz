This folder contains the files required to run the Flight Planning GUI created for Project Airwaze, undertaken by:

Jade Nassif (jade.nassif.385@cranfield.ac.uk)
Amey Shah (amey.shah.372@cranfield.ac.uk)
Aakalpa Deepak Savant (aakalpadeepak.savant.657@cranfield.ac.uk)
Swapnil Patil (swapnil.patil.897@cranfield.ac.uk)
Luca Mattiocco (luca.mattiocco@cranfield.ac.uk)
---------------
NOTE 1: THE GUI WAS OPTIMISED FOR A 1920x1080 SCREEN, IF YOU ATTEMPT TO LAUNCH IT ON A DIFFERENT RESOLUTION IT MAY NOT 
PRODUCE THE EXPECTED OUTPUT

NOTE 2: NO WI-FI CONNECTION IS REQUIRED TO RUN THE UI

NOTE 3: THE PRECIPITATION FIELD CAN BE MODIFIED. THIS CAN BE DONE FROM LINE 428 OF THE FlightPlanningUI.py FILE. INSTRUCTIONS ARE INCLUDED THERE. 
----------------
Running the GUI:

0. Ensure all required libraries are installed on your system. These may be found at the top of the class_definition.py and OWSAS.py files
   Some uncommon libraries may include:
   - PyQt5 (pip install PyQt5)
   - folium (pip install folium)
   - geographiclib (pip install geographiclib)
   - geopy (pip install geopy)
   - shapely (pip install shapely)
   - geopandas (pip install geopandas)
   - scikitlearn (pip install scikit-learn)
   
1. In the IDE of your choice, ensure the current directory is the folder containing the files

2. Open the FlightPlanningUI.py file and launch it. A window should show up and this is the GUI
-----------------
How to use the GUI:

The left of the UI is a workflow. 

1. The user enters flight details, this generates a map containing waypoints whose names can be obtained by clicking on each
2. After selected waypoints, the user enters a route defined by N waypoints as WYPT1-WYPT2-...-WYPTN from the departure to arrival airport.
3. This generates a map with a travel corridor, the user then has the choice to optimise the route for weather by clicking the "Optimise for Weather" button.
   A map displaying the original route in black along with the optimised route in red and the precipitation clusters is shown on the right.