# CPS_Project
Project developed for the final exam of Cyber Physical System's course.

The temperature control system is an example of a controller which is largely and daily used. This project is based on the study of an Arduino temperature micro-controller: Temperature Control Lab (TCLab) which is described in Park et al., 2020. The purpose of this project is to develop the controller. In order to reach the objective, a Proportional Integral Derivative (PID) controller has been developed. In addition, an Extended Kalman Filter has been studied to emulate the real sensor. Finally, verification and falsification are performed in Signal Temporary Language, using Moonlight library. The simulations were proposed using different values as the reference temperature.

Here is a brief summary of the contents of the files and folders.
Folders:
- Data: data obtained providing as input to the model the percentage of power of the heater; the parameters calulated for the FOPDT model; the parameters calculated for the PID controller;
- Performances: the performances measured in PID controller using different values as reference temperature;
- PID: data recorded during PID simulation;
- Plots: the plots of the PID controller developed using different values as reference temperature; the plot of the FOPDT model; two simulations carried out providing as input to the model the percentage of power of the heater;
- Requirements: the results of the verification of the requirements and of the falsification (using Moonlight library);
- CPS_Report.pdf: report for the exam; 
- efk_final.py: extended Kalman Filter;
- model_final.py: the model itself; all the functions used for the plots;
- performance.py: service functions;
- pid_final.py: the controller; the FOPDT model;
- requirements.py: verification and falsification using Moonlight (it can not be launched without it); 
- results.py: all the functions used for printing the results.
