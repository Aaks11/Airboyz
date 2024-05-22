#%%
from FlightRadar24 import FlightRadar24API
fr_api = FlightRadar24API()
flight_number_target="U24307"
zones=list((fr_api.get_zones()).keys())
flights=[]
aircraft_type = "B38M"
airline = "AXB"
i=0
# zone=fr_api.get_zones()["asia"]
# bounds = fr_api.get_bounds(zone)
# flights = fr_api.get_flights(aircraft_type = aircraft_type,airline = airline,bounds = bounds)
while flights==[]:
    zone=fr_api.get_zones()[zones[i]]
    bounds = fr_api.get_bounds(zone)
    flights = fr_api.get_flights(aircraft_type = aircraft_type,airline = airline,bounds = bounds)
    i=i+1
#%%s
for flight in flights:
    print(flight)
flight=flights[2]
flight_details = fr_api.get_flight_details(flight)
for key in flight_details.keys():
    print(key+ " : " + str(flight_details[key]))
# flight.set_flight_details(flight_details)

#get current position
lat=flight_details["trail"][0]["lat"]
lon=flight_details["trail"][0]["lng"]
altitude=flight_details["trail"][0]["alt"]
timestamp=flight_details["trail"][0]["ts"]
heading=flight_details["trail"][0]["hd"]
speed=flight_details["trail"][0]["spd"]
flight_number=flight_details["identification"]["number"]["default"]
#%%