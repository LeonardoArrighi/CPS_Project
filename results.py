import time
start_time=time.time()

from matplotlib.pyplot import savefig
from model_final import *
from pid_final import *
from ekf_final import *
from performance import *
import numpy as np
from os import path

# SET
out_path = "C:\\Users\\Leo\\Desktop\\Appunti\\CPS\\Project_2H\\DEF\\Plots"
pid_path = "C:\\Users\\Leo\\Desktop\\Appunti\\CPS\\Project_2H\\DEF\\PID"
data_path = "C:\\Users\\Leo\\Desktop\\Appunti\\CPS\\Project_2H\\DEF\\Data"
perf_path = "C:\\Users\\Leo\\Desktop\\Appunti\\CPS\\Project_2H\\DEF\\Performances"

show_all = False
current_version = 9
n = 901
mod_perf = False
need_it = True

print("start")

# service function
def set_t(n):
    t = np.linspace(0, n-1, n)
    return t

# universal variables
t = set_t(n)
ref = [303.15, 308.15, 313.15, 318.15, 323.15, 328.15, 333.15, 338.15, 343.15]
# ref = [313.15, 393.15]

if need_it:

    m = model()

    # EXAMPLES
    print("example")
    Q1, Q2 = m.set_heaters(t)
    u = (Q1, Q2)
    m.plot_result(u, t, flag='Other', save_this=None, save_plot=path.join(out_path,f'normal_{current_version}'), show_plot=show_all)

    print("just one heater")
    Q1, Q2 = m.set_heater1(t)
    u = (Q1, Q2)
    m.plot_result(u, t, flag='Other', save_this=data_path, version=current_version, save_plot=path.join(out_path,f'just_one_{current_version}'), show_plot=show_all)


    # FOPDT
    print("fopdt")
    data = np.loadtxt(path.join(data_path,f'data_{current_version}.txt'),delimiter=",",skiprows=1)
    # extract data columns
    t = data[:,0].T
    u0 = data[0,1]
    u = data[:,1].T
    yp0 = data[0,3]
    yp = data[:,3].T

    print("pid tuning")
    ns = len(t)
    delta_t = t[1]-t[0]
    uf = interp1d(t,u)
    p = pid(model, 0, 0, 0, True)
    p.tune_controller(u0, u, t, yp0, yp)
        
    p.plot_fopdt(t, u0, u, yp0, yp, save_plot=path.join(out_path,f'fopdt_{current_version}'), show_plot=show_all)


    print("kalman-filter")
    # K-F
    Q = np.diag([1e-2, 1e-2])  # covariance matrix for process noise
    R = np.diag([1e-2, 1e-2])  # covariance matrix for measurement noise
    P = np.diag([0.1, 0.1])  # initial state covariance matrix
    ekf = EKF(m, R, Q, P)


    print("pid")
    # PID
    with open(path.join(data_path,f'pid_parameters_{current_version}.txt'), "w") as txt_file:
        txt_file.write(f'Kc = {p.Kc}\ntau_d = {p.tau_d}\ntau_i = {p.tau_i}')

    pid_t = pid(m, Kc = p.Kc, tau_d = p.tau_d, tau_i = p.tau_i, observer = ekf)


    def print_results(ref, t, w_pid, version):
        with open(path.join(pid_path, f"pid_{ref-273.15}_{version}.txt"), "w") as txt_file:
            txt_file.write(f'{ref},C1,C2\n')
        # for i in range(len(ref)):
        T1, T2, C1, C2, Q1, Q2 = w_pid.pid(ref, t)
        m.plot_result((Q1, Q2), t, T1, T2, C1, C2, set=ref-273.15, flag='C', save_plot = path.join(out_path,f'{ref-273.15}_{version}'), show_plot=show_all)
        with open(path.join(pid_path, f"pid_{ref-273.15}_{version}.txt"), "a") as txt_file:
            for j in range(len(t)):
                txt_file.write(f'{ref},{C1[j]},{C2[j]}\n')

    for i in range(len(ref)):
        print(f"working on {ref[i]}")
        print_results(ref[i], t, pid_t, current_version)


if not need_it:
    t = set_t(n-1)
if mod_perf:
    print("performances")      
    for j in range(len(ref)):
        deg = ref[j] - 273.15
        res = np.loadtxt(path.join(pid_path,f'pid_{ref[j]-273.15}_{current_version}.txt'),delimiter=",",skiprows=1)
        
        with open(path.join(perf_path, f"performance_{ref[j]-273.15}_{current_version}.txt"), "w") as txt_file:
            txt_file.write(f"{ref[j]} K/ {deg}°C:\n")
            dict = {}
            C1 = res[:,1].T
            C2 = res[:,2].T
            os_1 = overshoot(C1, ref[j])  # overshoot
            os_2 = overshoot(C2, ref[j])  # overshoot
            dict["overshoot_H1"] = os_1
            dict["overshoot_H2"] = os_2
            rt_1 = rise_time(C1, t, ref[j])  # rise time
            rt_2 = rise_time(C2, t, ref[j])  # rise time
            dict["rise_time_H1"] = rt_1
            dict["rise_time_H2"] = rt_2
            sse_1 = steady_state_error(C1, ref[j])  # steady state error
            sse_2 = steady_state_error(C2, ref[j])  # steady state error
            dict["steady_state_error_H1"] = sse_1
            dict["steady_state_error_H2"] = sse_2
            st_1 = settling_time(C1, t)  # settling time
            st_2 = settling_time(C2, t)  # settling time
            dict["settling_time_H1"] = st_1
            dict["settling_time_H2"] = st_2
            txt_file.write(f"{dict}")

print("end")

end_time = time.time()
print("--- %s seconds ---" % (end_time - start_time))