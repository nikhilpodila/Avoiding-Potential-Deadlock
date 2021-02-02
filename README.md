# Avoiding Potential field deadlock
    Author: Nikhil Podila
    
## Code test environment:

The code was tested in Python (version 3.7.4) on Windows 10, installed using Anaconda distribution. 

The following are lines from the "requirements.txt" file of the project. <br>
It contains all the required dependencies/libraries/packages with the format: [package_name]==[version_number]
```
pybullet==2.1.3
numpy==1.16.5
```
## Getting started

### Avoiding potential field deadlock
This is the <b>main task</b> of the assignment. The algorithm is detailed in the documentation

Results and simulation can be viewed by running the following command on the command prompt or terminal in this directory:
```
python experiment.py
```

### Debug trail lines
This is an <b>extra feature</b> to view:
1. Trajectory followed by the Robot's End Effector to perform its tasks (red line)
2. Instantaneous velocity (Tangent) direction calculated by the Boundary Following algorithm to move around the obstacle (blue line)

This feature can be used by adding ```debug``` as an argument. That is, run the following command on the command prompt or terminal in this directory:
```
python experiment.py debug
```
### Viewing the potential field deadlock
This is an <b>extra feature</b> to view the deadlock caused on the robot due to Obstacle and Target when using Total Potential field method for Planning. The explanation and details are provided in the documentation.

This feature can be used by adding ```repulsive``` as an argument. That is, run the following command on the command prompt or terminal in this directory:
```
python experiment.py repulsive
```
or
```
python experiment.py repulsive debug
```