# NOTE_: this file must be contained in the distribution/python directory of Moonlight

from random import sample
from sys import path
path.append("C:\\Users\\Leo\\Desktop\\Appunti\\CPS\\Project_2H\\DEF")
from results import *
from moonlight import *
from random import uniform
import re

# Set
req_path = "C:\\Users\\Leo\\Desktop\\Appunti\\CPS\\Project_2H\\DEF\\Requirements"

def oscillations(C):
    oo = [abs(C[i] - C[i-1]) for i in range(1, len(C))]
    return oo

def proximity(C, r):
    pp = [abs(C[i] - r[i]) for i in range(1, len(C))] 
    return pp
    
# requirements
script_oscillations = """
signal {real o;}
domain minmax;
formula initial_oscillation = {eventually {globally [0.0, 300.0] (o < 1.0)}};
formula final_oscillation = {globally [300.0, 900.0] (o < 0.5)};
"""

script_proximity = """
signal {real p;}
domain minmax;
formula proximity = {eventually {globally [300.0, 900.0] (p < 1.0)}};
"""


script_max = """
signal {real m;}
domain minmax;
formula maxx = {globally [1.0, 900.0] (m < 373.15)};
"""


moonlightScript_o = ScriptLoader.loadFromText(script_oscillations)
monitor_1 = moonlightScript_o.getMonitor("initial_oscillation")
monitor_2 = moonlightScript_o.getMonitor("final_oscillation")
moonlightScript_p = ScriptLoader.loadFromText(script_proximity)
monitor_3 = moonlightScript_p.getMonitor("proximity")
moonlightScript_m = ScriptLoader.loadFromText(script_max)
monitor_4 = moonlightScript_m.getMonitor("maxx")

# def test_functions(ref, n, version):
#     for j in range(len(ref)):
#         t = set_t(n-1)
#         res = np.loadtxt(path.join(pid_path,f'pid_{ref[j]-273.15}_{version}.txt'),delimiter=",",skiprows=1)
#         r = res[:,0].T
#         C1 = res[:,1].T
#         C2 = res[:,2].T
#         print(ref[j])
#         p_1 = [[pp] for pp in proximity(C1, r)]
#         print(p_1)
# test_functions(ref, n, current_version)

def requirement(ref, n, version):
    for j in range(len(ref)):
        t = set_t(n-1)
        deg = ref[j] - 273.15
        res = np.loadtxt(path.join(pid_path,f'pid_{ref[j]-273.15}_{version}.txt'),delimiter=",",skiprows=1)
        with open(path.join(req_path, f"requirements_{ref[j]-273.15}_{version}.txt"), "w") as txt_file:
            # txt_file.write(f"{ref[j]} K/ {deg}°C:\n")
            r = res[:,0].T
            C1 = res[:,1].T
            C2 = res[:,2].T

            dict = {}
            
            o_1 = [[oo] for oo in oscillations(C1)]
            o_2 = [[oo] for oo in oscillations(C2)]
            result_init_o_H1 = monitor_1.monitor(list(t[:-1]), o_1)
            result_init_o_H2 = monitor_1.monitor(list(t[:-1]), o_2)
            dict["initial_oscillation_H1"] = result_init_o_H1[0][1]
            dict["initial_oscillation_H2"] = result_init_o_H2[0][1]
            result_final_o_H1 = monitor_2.monitor(list(t[:-1]), o_1)
            result_final_o_H2 = monitor_2.monitor(list(t[:-1]), o_2)
            dict["final_oscillation_H1"] = result_final_o_H1[0][1]
            dict["final_oscillation_H2"] = result_final_o_H2[0][1]
            
            p_1 = [[pp] for pp in proximity(C1, r)]
            p_2 = [[pp] for pp in proximity(C2, r)]
            result_prox_H1 = monitor_3.monitor(list(t[:-1]), p_1)
            result_prox_H2 = monitor_3.monitor(list(t[:-1]), p_2)
            dict["proximity_H1"] = result_prox_H1[0][1]
            dict["proximity_H2"] = result_prox_H2[0][1]
            
            c_1 = [[c] for c in C1]
            c_2 = [[c] for c in C2]
            result_max_H1 = monitor_4.monitor(list(t), c_1)
            result_max_H2 = monitor_4.monitor(list(t), c_2)
            dict["max_H1"] = result_max_H1[0][1]
            dict["max_H2"] = result_max_H2[0][1]
            
            txt_file.write(f"{dict}")

def falsification(ref_i, n, n_simulations, version):
    print(f'working on {ref_i}')
    t = set_t(n-1)
    new_m = model()
    parameters = []
    with open (path.join(data_path,f'pid_parameters_{version}.txt'), "r") as txt_file:
        for line in txt_file:
            data = line.split()
            parameters.append(float(data[2]))
    min_f_1 = float('Inf')
    min_f_2 = float('Inf')
    with open(path.join(req_path, f"falsification_{ref_i-273.15}_{version}.txt"), "w") as txt_file:    
        
        dict = {}
        for i in range(n_simulations):    

            Q = np.diag([uniform(0.01,0.0001),uniform(0.01,0.0001)])  # covariance matrix for process noise
            R = np.diag([uniform(0.01,0.0001),uniform(0.01,0.0001)])  # covariance matrix for measurement noise
            P = np.diag([0.1, 0.1])  # initial state covariance matrix
            ekf = EKF(new_m, R, Q, P)
            
            new_pid = pid(new_m, Kc = parameters[0], tau_d = parameters[1], tau_i = parameters[2], observer = ekf)
            T1, T2, C1, C2, Q1, Q2 = new_pid.pid(ref_i, t)
            
            c_1 = [[c] for c in C1]
            c_2 = [[c] for c in C2]
            result_H1 = monitor_4.monitor(list(t), c_1)[0][1]
            result_H2 = monitor_4.monitor(list(t), c_2)[0][1]
            if result_H1 < min_f_1:
                min_f_1 = result_H1
            if result_H2 < min_f_2:
                min_f_2 = result_H2
            if min_f_1 < 0:
                min_f_1 = -1
            if min_f_2 < 0:
                min_f_2 = -1
            dict[f"fals_max_H1"] = min_f_1
            dict[f"fals_max_H2"] = min_f_2

        txt_file.write(f"{dict}")
        

# print('start requirements')
# requirement(ref, n, current_version)

print('start falsification')
for i in range(len(ref)):
    falsification(ref[i], n, 10, current_version)

req_time = time.time()
print("--- %s seconds ---" % (req_time - start_time))