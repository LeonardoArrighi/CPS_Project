from scipy.optimize import minimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np
from model_final import *

data_path = "C:\\Users\\Leo\\Desktop\\Appunti\\CPS\\Project_2H\\DEF\\Data"

class pid:

    def __init__(self, model, Kc=None, tau_i=None, tau_d=None, d=True, observer=None):
        self.model = model
        self.id = None  
        self.Kc = Kc
        self.tau_i = tau_i
        self.d = d  
        if self.d:
            self.tau_d = tau_d
        self.observer = observer # ekf

    def fopdt(self, y, t, x, u0, uf, yp0):
        Km, taum, thetam = x
        try:
            if (t - thetam) <= 0:
                um = uf(0.0)
            else:
                um = uf(t - thetam)
        except:
            um = u0
        dydt = (-(y - yp0) + Km * (um - u0)) / taum
        return dydt

    def sim_fopdt(self, x, u0, uf, t, yp0):
        Km, taum, thetam = x
        dt = t[1] - t[0]
        ym = np.zeros(len(t))  # storage for model values
        ym[0] = yp0  # initial condition
        for i in range(len(t) - 1):
            ts = [dt * i, dt * (i + 1)]
            y1 = odeint(self.fopdt, ym[i], ts, args=(x, u0, uf, yp0))
            ym[i + 1] = y1[-1]
        return ym

    def objective(self, x, u0, uf, t, yp0, yp):
        ym = self.sim_fopdt(x, u0, uf, t, yp0)
        # calculate objective
        obj = 0.0
        for i in range(len(ym)):
            obj = obj + (ym[i] - yp[i]) ** 2 # SSE
        return obj

    def identification(self, x0, bounds, u0, u, t, yp0, yp):
        uf = interp1d(t, u)
        solution = minimize(self.objective, x0, args=(u0, uf, t, yp0, yp), method="SLSQP", bounds=bounds)
        self.id = solution.x  # [K, tau, theta]

    def plot_fopdt(self, t, u0, u, yp0, yp, save_plot=None, show_plot=True):
        assert (self.id is not None)
        uf = interp1d(t, u)
        x0 = np.array([0.25, 200.0, 0.0])
        sim1 = self.sim_fopdt(x0, u0, uf, t, yp0)
        sim2 = self.sim_fopdt(self.id, u0, uf, t, yp0)
        plt.figure(1, figsize=(15, 8))
        plt.plot(t, sim1, 'r--', label="Guessed")
        plt.plot(t, sim2, 'r-', label="Approximation")
        plt.plot(t, yp, 'b-', label="Output")
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Temperature $^\circ C$')
        if save_plot != None:
            plt.savefig(f'{save_plot}.png')
        if show_plot:
            plt.show()


    def imc(self):
        assert (self.id is not None)
        lambd = max(0.1*self.id[1], 0.8*self.id[2])
        self.Kc = (1/self.id[0]) * ((self.id[1] + 0.5 * self.id[2])/(lambd + 0.5 * self.id[2]))
        self.tau_i = self.id[1] + 0.5 * self.id[2]
        if self.d:
            self.tau_d = ((self.id[1]*self.id[2])/(2*self.id[1] + self.id[2]))

    def tune_controller(self, u0, u, t, yp0, yp): # find pid parameteres
        x0 = np.array([0.25, 200.0, 0.0])  # initial guess
        uf = interp1d(t, u)
        # print('Initial SSE Objective: ' + str(self.objective(x0, u0, uf, t, yp0, yp)))
        bounds = ((-1.0e10, 1.0e10), (0.01, 1.0e10), (0.0, 1.0e10))
        self.identification(x0, bounds, u0, u, t, yp0, yp)
        # print('Kp: ' + str(self.id[0]))
        # print('taup: ' + str(self.id[1]))
        # print('thetap: ' + str(self.id[2]))
        # print('Final SSE Objective: ' + str(self.objective(self.id, u0, uf, t, yp0, yp)))
        with open(path.join(data_path,'fopdt_parameters.txt'), "w") as txt_file:
            txt_file.write(f'Kp = {self.id[0]}\ntau_p = {self.id[1]}\ntheta_p = {self.id[2]}\nFinal SSE Objective: {self.objective(self.id, u0, uf, t, yp0, yp)}')
        self.imc()

    def pid(self, ref, t):
        assert (self.Kc is not None)
        assert (self.tau_i is not None)
        if self.d:
            assert(self.tau_d is not None)
        # Heater 1
        out1 = np.zeros(len(t))  # controller output
        proc1 = np.zeros(len(t))  # controlled variable
        e1 = np.zeros(len(t))  # error
        ig1 = np.zeros(len(t))  # integral of the error
        if self.d:
            d1 = np.zeros(len(t))  # derivative of the error
            D1 = np.zeros(len(t))
        P1 = np.zeros(len(t))
        I1 = np.zeros(len(t))
        # Heater 2
        out2 = np.zeros(len(t))  # controller output
        proc2 = np.zeros(len(t))  # controlled variable
        e2 = np.zeros(len(t))  # error
        ig2 = np.zeros(len(t))  # integral of the error
        if self.d:
            d2 = np.zeros(len(t))  # derivative of the error
            D2 = np.zeros(len(t))
        P2 = np.zeros(len(t))
        I2 = np.zeros(len(t))        
        # reactor variables
        T1 = np.ones(len(t)) * self.model.T_inf
        T2 = np.ones(len(t)) * self.model.T_inf
        C1 = np.ones(len(t)) * self.model.T_inf
        C2 = np.ones(len(t)) * self.model.T_inf
        # PID loop
        u1 = np.ones(len(t))
        u2 = np.ones(len(t))
        x0 = [self.model.T_inf, self.model.T_inf, self.model.T_inf, self.model.T_inf]  # initial conditions
        for i in range(len(t)-1):
            dt = t[i+1] - t[i]  # current time-step
            # Heater 1
            e1[i] = ref - proc1[i]  # current step error
            if i > 0:
                ig1[i] = ig1[i-1] + e1[i] * dt  # discrete integral
                if self.d:
                    d1[i] = (e1[i] - e1[i-1])/dt  # discrete derivative
            P1[i] = self.Kc * e1[i]  # proportional term
            I1[i] = self.Kc/self.tau_i * ig1[i]  # integral term
            if self.d:
                D1[i] = self.tau_d * d1[i]
            out1[i] = P1[i] + I1[i] if not self.d else P1[i] + I1[i] + D1[i]
            
            # Heater 2
            e2[i] = ref - proc2[i]  # current step error
            if i > 0:
                ig2[i] = ig2[i-1] + e2[i] * dt  # discrete integral
                if self.d:
                    d2[i] = (e2[i] - e2[i-1])/dt  # discrete derivative
            P2[i] = self.Kc * e2[i]  # proportional term
            I2[i] = self.Kc/self.tau_i * ig2[i]  # integral term
            if self.d:
                D2[i] = self.tau_d * d2[i]
            out2[i] = P2[i] + I2[i] if not self.d else P2[i] + I2[i] + D2[i]

            # bounds for the controlled variable
            if (out1[i] > 100):
                out1[i] = 100
            if (out2[i] > 100):
                out2[i] = 100
            if (out1[i] < 0):
                out1[i] = 0
            if (out2[i] < 0):
                out2[i] = 0

            ts = [t[i], t[i+1]]  # current time step
            u1[i] = out1[i]
            u2[i] = out2[i]
            u = (u1[i], u2[i])
            x0 = self.model.sim_step(x0, u, ts, T1, T2, C1, C2, i)
            # adding noise to the output
            T1[i + 1] += np.random.normal(0, 0.1)
            T2[i + 1] += np.random.normal(0, 0.1)
            C1[i + 1] += np.random.normal(0, 0.1)
            C2[i + 1] += np.random.normal(0, 0.1)
            # ekf
            if self.observer is not None:
                T1[i+1], C1[i+1] = self.observer.observe([T1[i+1], C1[i+1]], u1[i], np.eye(2), dt, 'H1')
                T2[i+1], C2[i+1] = self.observer.observe([T2[i+1], C2[i+1]], u2[i], np.eye(2), dt, 'H2')
            proc1[i+1] = C1[i+1]
            proc2[i+1] = C2[i+1]
        return T1, T2, C1, C2, u1, u2


