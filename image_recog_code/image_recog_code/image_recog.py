
'''
   +---------------------------------------------------+
   | Code Developed by Amey Shah                       |
   | Aerospace Computational Engineering MSc 2023-2024 |
   | Cranfield University                              |
   | amey.shah.372@cranfield.ac.uk                     |
   +---------------------------------------------------+ 
'''

import cv2
import numpy as np
import csv
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# Seting up Selenium webdriver
chrome_options = Options()
chrome_options.add_argument("--headless") 
driver = webdriver.Chrome(options=chrome_options)

# Loading the HTML file as server not a file from folder
driver.get("http://127.0.0.1:5500/squaremap_with_onebutton.html")
#http://127.0.0.1:5500/squaremap_with_onebutton.html

# Geting the map container element from the driver.
body = driver.find_element(By.ID, 'map-container')
map_container = body.find_element(By.ID, "map")

# Tkaing a screenshot of the map container
map_container.screenshot("weather_map.png")

# Close the server
driver.quit()

# Load the screenshot
img = cv2.imread('weather_map.png')
height, width, _ = img.shape

# Converting the image to HSV color space
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define the range of yellow pixels in HSV
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

# Creating a mask for yellow pixels
mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)

# Finding the coordinates of yellow pixels
yellow_pixels = np.column_stack(np.where(mask == 255))

# Creating a blank image for plotting yellow pixels
yellow_pixel_map = np.zeros_like(img)

# Ploting and saving the yellow pixels on the blank image
for x, y in yellow_pixels:
    if y < height and x < width:
        yellow_pixel_map[x, y] = (0, 255, 255)  # Set yellow color (BGR)
cv2.imwrite('yellow_pixel_map.png', yellow_pixel_map)

# Create a CSV file with yellow pixel coordinates coridnates. 
with open('yellow_pixel_coordinates.csv', 'w', newline='') as csvfile:
    fieldnames = ['Latitude', 'Longitude']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()

    # Using the pixel size and the center latitude and longitude the neighbouring cordinates are calculated and saved into CSV. 
    center_lat, center_lon = 48.55979167659633, 12.920652953465245
    zoom_level = 9
    tile_size = 512
    max_tiles = 1 ** zoom_level

    for x, y in yellow_pixels:
        tile_x = x // tile_size
        tile_y = y // tile_size
        tile_lon = center_lon + (tile_x - max_tiles // 2) * 360 / max_tiles
        tile_lat = center_lat - (tile_y - max_tiles // 2) * 180 / max_tiles

        pixel_x = x % tile_size
        pixel_y = y % tile_size
        pixel_lon = tile_lon + (pixel_x / tile_size - 0.5) * 360 / max_tiles
        pixel_lat = tile_lat + (0.5 - pixel_y / tile_size ) * 180 / max_tiles

        writer.writerow({'Latitude': pixel_lat, 'Longitude': pixel_lon})