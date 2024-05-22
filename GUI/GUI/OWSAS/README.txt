This folder contains the files required to run the OWSAS GUI created for Project Airwaze, undertaken by:

Jade Nassif (jade.nassif.385@cranfield.ac.uk)
Amey Shah (amey.shah.372@cranfield.ac.uk)
Aakalpa Deepak Savant (aakalpadeepak.savant.657@cranfield.ac.uk)
Swapnil Patil (swapnil.patil.897@cranfield.ac.uk)
Luca Mattiocco (luca.mattiocco@cranfield.ac.uk)
---------------
NOTE 1: THE GUI WAS OPTIMISED FOR A 1920x1080 SCREEN, IF YOU ATTEMPT TO LAUNCH IT ON A DIFFERENT RESOLUTION IT MAY NOT 
PRODUCE THE EXPECTED OUTPUT

NOTE 2: A WI-FI CONNECTION IS REQUIRED TO RUN THE UI

NOTE 3: The user is expected to create their own API keys for the following APIs:

* Current API at Openweathermap.org, free of charge
* CheckWX.com, free of charge

ONCE THIS IS DONE, THE user is expected to update the API key in lines
- 187
- 480
of the class_definition.py file
----------------
Running the GUI:

0. Ensure all required libraries are installed on your system. These may be found at the top of the class_definition.py and OWSAS.py files
   Some uncommon libraries may include:
   - PyQt5 (pip install PyQt5)
   - folium (pip install folium)
   - FlightRadar24 (pip install FlightRadarAPI)
   - geopy (pip install geopy)
   - pygeodesy (pip install PyGeodesy)
   - pyproj (pip install pyproj)
   
1. In the IDE of your choice, ensure the current directory is the folder containing the files

2. Open the OWSAS.py file and launch it. A window should show up and this is the GUI
-----------------
How to use the GUI:

1. The user is expected to enter some flight details in the menu on the left. 
    - Target flight number
    - The aircraft's ICAO code (model number, eg. B737 for a Boeing 737)
    - The airline's ICAO code  (eg. RYR for Ryanair or ETD for Etihad Airways)

This HAS to be an ongoing flight and the user may check this through FlightRadar24.com
The user then submits the flight details.

Alternatively, the user may choose to upload a predefined trail. In this case, the file must be a 
.csv file with the following format

lat | lon | alt | timestamp | heading | ground speed 

Where the first entry is the latest timestamp and the last is the first timestamp recorded.
Units are feet for altitude, timestamp in Unix format (number of seconds since January 1, 1970), heading in degrees, ground speed in knots


2. Once step 1 is done, the user enters a radius of interest around the aircraft in kilometers, and this is kept constant throughout the flight
The user then hits "Start/Refresh".
The GUI will then obtain the required weather information and display it

3. Refresh the information when desired. It is recommended to not do this too frequently due to the limit on API calls.