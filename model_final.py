import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import csv
from os import path

class model:

  def __init__(self):
    '''Parameters decalaration'''
    self.T_inf = 296.15      # K
    self.T_inf_deg = 23.00   # ° C
    self.T_start = 296.15    # K
    self.T_start_deg = 23.00 # ° C
    self.alpha1 = 0.01       # W / % heater 1
    self.alpha2 = 0.0075     # W / % heater 2
    self.Cp = 500            # J/(kg*K)
    self.A = 1e-3            # m^2
    self.As = 2e-4           # m^2
    self.m = 0.004           # kg
    self.eps = 0.9           
    self.U = 4.4             # W/(m^2*K)
    self.Us = 24.0           # W/(m^2*K)
    self.sigma = 5.67e-8 
    # self.tau_c_low = 21.1    # s
    self.tau_c_up = 23.3     # s

  def C12(self, x):
    '''Heat transfer exchange: conduction'''
    T1, T2, C1, C2 = x
    c12 = self.Us * self.As * (T2 - T1)
    return c12
  
  def R12(self, x):
    '''Heat transfer exchange: radiation'''
    T1, T2, C1, C2 = x
    r12 = self.eps * self.sigma * self.As * (T2**4 - T1**4)
    return r12
  
  def set_heaters(self, t):
    '''Decide the setting of the heater'''
    Q1 = np.zeros(len(t))
    Q2 = np.zeros(len(t))
    
    Q1[0:] = 0.0 
    Q1[10:] = 80.0
    Q1[200:] = 20.0
    Q1[280:] = 70.0
    Q1[400:] = 50
    Q2[0:] = 0.0 
    Q2[120:] = 100.0
    Q2[320:] = 10.0
    Q2[520:] = 80.0
        
    return Q1, Q2
  
  def set_heater1(self, t):
    '''Decide the setting of the heater 1, while heater 2 remains off'''
    Q1 = np.zeros(len(t))
    Q2 = np.zeros(len(t))
    
    Q1[0:] = 0.0 
    Q1[10:] = 80.0
    Q1[200:] = 20.0
    Q1[280:] = 70.0
    Q1[400:] = 50

    return Q1, Q2


  def odes(self, x, t, u):
    '''Equations'''
    T1, T2, C1, C2 = x
    Q1, Q2 = u
    dT1dt = (1.0 / (self.m * self.Cp))*(self.U * self.A * (self.T_inf - T1) \
                                        + self.eps * self.sigma * self.A * (self.T_inf**4 - T1**4) \
                                        + self.C12(x) + self.R12(x) + self.alpha1 * Q1)
    dT2dt = (1.0 / (self.m * self.Cp))*(self.U * self.A * (self.T_inf - T2) \
                                        + self.eps * self.sigma * self.A * (self.T_inf**4 - T2**4) \
                                        - self.C12(x) - self.R12(x) + self.alpha2 * Q2)
    dC1dt = (T1 - C1)/self.tau_c_up
    dC2dt = (T2 - C2)/self.tau_c_up
    return [dT1dt, dT2dt, dC1dt, dC2dt]
  
  # def odes_Q2off(self, x, t, u):
  #   '''Equations'''
  #   T1, T2, C1, C2 = x
  #   Q1 = u
  #   Q2 = 0
  #   dT1dt = (1.0 / (self.m * self.Cp))*(self.U * self.A * (self.T_inf - T1) \
  #                                       + self.eps * self.sigma * self.A * (self.T_inf**4 - T1**4) \
  #                                       + self.C12(x) + self.R12(x) + self.alpha1 * Q1)
  #   dT2dt = (1.0 / (self.m * self.Cp))*(self.U * self.A * (self.T_inf - T2) \
  #                                       + self.eps * self.sigma * self.A * (self.T_inf**4 - T2**4) \
  #                                       - self.C12(x) - self.R12(x) + self.alpha2 * Q2)
  #   dC1dt = (T1 - C1)/self.tau_c_up
  #   dC2dt = (T2 - C2)/self.tau_c_up
  #   return [dT1dt, dT2dt, dC1dt, dC2dt]

  def discrete_step(self, x, dt, u, T1, T2, C1, C2, i):
    dT1dt, dT2dt, dC1dt, dC2dt = self.odes(x, dt, u)
    T1[i+1] = T1[i] + dT1dt*dt
    T2[i+1] = T2[i] + dT2dt*dt
    C1[i+1] = C1[i] + dC1dt*dt
    C2[i+1] = C2[i] + dC2dt*dt
    x = T1[i+1], T2[i+1], C1[i+1], C2[i+1]
    return x
  
  def simple_step(self, x, u, dt, flag):
    T, C = x
    if flag == 'H1':
      x = T, self.T_start, C, self.T_start
      uu = (u, 0)
      
    elif flag == 'H2':
      x = self.T_start, T, self.T_start, C
      uu = (0, u)
    dT1dt, dT2dt, dC1dt, dC2dt = self.odes(x, dt, uu)
    T1 = x[0] + dT1dt*dt
    T2 = x[1] + dT2dt*dt
    C1 = x[2] + dC1dt*dt
    C2 = x[3] + dC2dt*dt
    if flag == 'H1':
      T_f = T1
      C_f = C1
    elif flag == 'H2':
      T_f = T2
      C_f = C2
    return T_f, C_f
  
  def discrete_sim(self, u, t):
    x = [self.T_start, self.T_start, self.T_start, self.T_start]
    T1 = np.ones(len(t)) * x[0]
    T2 = np.ones(len(t)) * x[1]
    C1 = np.ones(len(t)) * x[2]
    C2 = np.ones(len(t)) * x[3]
    Q1, Q2 = u
    for i in range(len(t)-1):
      dt = t[i+1] - t[i]
      u = (Q1[i], Q2[i])
      x = self.discrete_step(x, dt, u, T1, T2, C1, C2, i)
    return T1, T2, C1, C2

  def sim_step(self, x, u, ts, T1, T2, C1, C2, i):
    y = odeint(self.odes, x, ts, args=(u,))
    T1[i+1], T2[i+1], C1[i+1], C2[i+1] = y[1]
    x = T1[i+1], T2[i+1], C1[i+1], C2[i+1]
    return x
  
  # def sim_step_Q2off(self, x, u, ts, T1, T2, C1, C2, i):
  #   y = odeint(self.odes_Q2off, x, ts, args=(u,))
  #   T1[i+1], T2[i+1], C1[i+1], C2[i+1] = y[1]
  #   x = T1[i+1], T2[i+1], C1[i+1], C2[i+1]
  #   return x
  
  def simulate(self, u, t, data_path, version):
    if data_path != None:
      with open(path.join(data_path,f'data_{version}.csv'), "w", newline='') as csv_file:
        obj_write = csv.writer(csv_file)
        obj_write.writerow(['Time (sec)','Q1','Q2','T1','T2','Set Point 1 (degC)','Set Point 2 (degC)'])
      with open(path.join(data_path,f'data_{version}.txt'), "w") as txt_file:
        txt_file.write('Time (sec),Q1,Q2,T1,T2,Set Point 1 (degC),Set Point 2 (degC)\n')
      formatter = "{0:.8f}"
    
    x = [self.T_start,self.T_start,self.T_start,self.T_start]
    T1 = np.ones(len(t)) * x[0]
    T2 = np.ones(len(t)) * x[1]
    C1 = np.ones(len(t)) * x[2]
    C2 = np.ones(len(t)) * x[3]
    Q1, Q2 = u
    for i in range(len(t) - 1):
      ts = [t[i], t[i+1]]
      u = (Q1[i], Q2[i])
      x = self.sim_step(x, u, ts, T1, T2, C1, C2, i)
      if data_path != None:
        with open(path.join(data_path,f'data_{version}.csv'), "a", newline='') as csv_file:
          obj_write = csv.writer(csv_file)
          obj_write.writerow([f'{formatter.format(t[i])}',f'{formatter.format(Q1[i])}',f'{formatter.format(Q2[i])}',f'{formatter.format(C1[i]-273.15)}',f'{formatter.format(C2[i]-273.15)}',f'{formatter.format(self.T_inf_deg)}',f'{formatter.format(self.T_inf_deg)}'])
        with open(path.join(data_path,f'data_{version}.txt'), "a") as txt_file:
          txt_file.write(f"{formatter.format(t[i])},{formatter.format(Q1[i])},{formatter.format(Q2[i])},{formatter.format(C1[i]-273.15)},{formatter.format(C2[i]-273.15)},{formatter.format(self.T_inf_deg)},{formatter.format(self.T_inf_deg)}\n")
    return T1, T2, C1, C2

  # def simulate_Q2off(self, u, t):
  #   x = [self.T_start,self.T_start,self.T_start,self.T_start]
  #   T1 = np.ones(len(t)) * x[0]
  #   T2 = np.ones(len(t)) * x[1]
  #   C1 = np.ones(len(t)) * x[2]
  #   C2 = np.ones(len(t)) * x[3]
  #   for i in range(len(t) - 1):
  #     ts = [t[i], t[i+1]]
  #     x = self.sim_step_Q2off(x, u[i], ts, T1, T2, C1, C2, i)
  #   return T1, T2, C1, C2

# plot_Q2_on = True,
  def plot_result(self, u, t, T1=np.zeros(1), T2=np.zeros(1), C1=np.zeros(1), C2=np.zeros(1), flag = 'Both', set = None, save_this = None, version = None, save_plot = None, show_plot = True):
    # if plot_Q2_on:
    #   Q1, Q2 = self.set_heaters(t)
    # else:
    #   Q1, Q2 = self.set_heater1(t)
    #   # Q2 = np.zeros(len(t))
    #   # u = Q1
    #   # T1, T2, C1, C2 = self.simulate_Q2off(u, t)
    if (T1[0] == 0 and T2[0] == 0 and C1[0] == 0 and C2[0] == 0):
      T1, T2, C1, C2 = self.simulate(u, t, save_this, version)
    Q1, Q2 = u

    plt.figure(1, figsize=(15, 8))
    
    # plot T
    plt.subplot(2,1,1)
    if (flag=='T'):
      plt.plot(t, T1-273.15, 'b-', label=r'$T_1$')
      plt.plot(t, T2-273.15, 'r-', label=r'$T_2$')
    elif (flag=='C'):
      plt.plot(t, C1-273.15, 'b-', label=r'$C_1$')
      plt.plot(t, C2-273.15, 'r-', label=r'$C_2$') 
    else:
      plt.plot(t, C1-273.15, 'r-', linewidth=3.0, label=r'$C_1$')
      plt.plot(t, C2-273.15, 'b-', linewidth=3.0, label=r'$C_2$')
      plt.plot(t, T1-273.15, 'r--', label=r'$T_1$')
      plt.plot(t, T2-273.15, 'b--', label=r'$T_2$')
    
    if set is not None:
        s = [set for i in range(len(t))]
        plt.plot(t, s, 'r:', label='Set point')

    plt.ylabel('Temperature $^\circ C$')
    plt.legend(loc='best')

    # plot Q
    plt.subplot(2,1,2)
    plt.plot(t, Q1, 'r-', label=r'$Q_1$')
    plt.plot(t, Q2, 'b-', label=r'$Q_2$')
    plt.ylabel('Heater Output')
    plt.legend(loc='best')
    
    plt.xlabel('Time (s)')


    if show_plot:
      plt.show()
    if save_plot != None:
      plt.savefig(f'{save_plot}.png')
    
    plt.clf()

# m = model()
# n = 601
# time = np.linspace(0, n-1, n)
# Q1, Q2 = m.set_heaters(time)
# u = (Q1, Q2)
# m.plot_result(u, time, flag='Other', set = 35.0, show_plot=False, save_plot='peto')

# Q1, Q2 = m.set_heater1(time)
# u = (Q1, Q2)
# m.plot_result(u, time, flag='Other', save_this=True)
