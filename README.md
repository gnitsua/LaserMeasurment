# LaserMeasurment

This repository contains code to help measure angular displacment using laser projection. Please see the attached paper for the experimental set up.

## Installation

Please pip install `requirments.txt` for the Python enviroment in which you will be running the code.

## Execution

The program takes a few command line arguments.

| Argument  | Value |
| ------------- | ------------- |
| --input  | video file to load  |
| --calibration  | numpy file containing the calibration as produced by opencv  |
| --name  | does nothing right now  |

## Output
The program will output a .png file plotting the height of the contour over time. 
It will also output statistics on the average height and width, as well as their standard deviations.
