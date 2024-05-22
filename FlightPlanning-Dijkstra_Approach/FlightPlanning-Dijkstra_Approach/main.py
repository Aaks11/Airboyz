################################################################################
#                                                                              #
#                          Flight Planning Code                                #
#                    using Dijkstra's Approach                                 #
#                                                                              #
# Author: [Luca Mattiocco]                                                     #
# Date: April 2024                                                             #
# Affiliation: Cranfield University                                            #
#                                                                              #
# Description:                                                                 #
# This code implements a flight planning algorithm using Dijkstra's approach   #
# to find the shortest path between two airports based on given flight         #
# connections and distances.                                                   #
#                                                                              #
#                                                                              #
#                                                                              #
################################################################################


import subprocess
import webbrowser

def run_script(script_path):
    try:
        subprocess.run(["python", script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing script {script_path}: {e}")

def open_html(html_file):
    webbrowser.open(html_file)

def main():
    # List of paths of the scripts to run
    scripts_to_run = [
        "AirportSelection.py",
        "FlightPathFindingCode.py"
    ]
    
    # Execute each script in the list
    for script_path in scripts_to_run:
        run_script(script_path)

    # Open the HTML file after executing the scripts
    open_html("Waypoints_Cluster_Connections_Delaunay_with_names_and_shortest_path_and_visited.html")

if __name__ == "__main__":
    main()
