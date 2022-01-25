import numpy as np

from model_final import *
import sympy as sp
import copy


class EKF:

    def __init__(self, model, R, Q, P):
        self.model = model
        self.A1 = self.get_matrices_H1()[0]
        self.B1 = self.get_matrices_H1()[1]
        self.A2 = self.get_matrices_H2()[0]
        self.B2 = self.get_matrices_H2()[1]
        self.Q_k = Q
        self.R_k = R
        self.P_k_1 = P

    def symbolic_diff_H1(self):
        T1 = sp.Symbol('T1')
        T2 = sp.Symbol('T2')
        C1 = sp.Symbol('C1')
        C2 = sp.Symbol('C2')
        Q1 = sp.Symbol('Q1')
        Q2 = sp.Symbol('Q2')
        x = T1, T2, C1, C2
        
        dT1dt = (1.0 / (self.model.m * self.model.Cp))*(self.model.U * self.model.A * (self.model.T_inf - T1) \
                                        + self.model.eps * self.model.sigma * self.model.A * (self.model.T_inf**4 - T1**4) \
                                        + self.model.C12(x) + self.model.R12(x) + self.model.alpha1 * Q1)
        dC1dt = (T1 - C1)/self.model.tau_c_up
            
        df1dT1 = dT1dt.diff(T1)
        df1dC1 = dT1dt.diff(C1)
        df2dC1 = dC1dt.diff(C1)
        df2dT1 = dC1dt.diff(T1)
        df2du1 = dT1dt.diff(Q1)
        
        return [df1dT1, df1dC1, df2dC1, df2dT1, df2du1]

    def symbolic_diff_H2(self):
        T1 = sp.Symbol('T1')
        T2 = sp.Symbol('T2')
        C1 = sp.Symbol('C1')
        C2 = sp.Symbol('C2')
        Q1 = sp.Symbol('Q1')
        Q2 = sp.Symbol('Q2')
        x = T1, T2, C1, C2
        
        dT2dt = (1.0 / (self.model.m * self.model.Cp))*(self.model.U * self.model.A * (self.model.T_inf - T2) \
                                            + self.model.eps * self.model.sigma * self.model.A * (self.model.T_inf**4 - T2**4) \
                                            - self.model.C12(x) - self.model.R12(x) + self.model.alpha2 * Q2)
        dC2dt = (T2 - C2)/self.model.tau_c_up
            
        df1dT2 = dT2dt.diff(T2)
        df1dC2 = dT2dt.diff(C2)
        df2dC2 = dC2dt.diff(C2)
        df2dT2 = dC2dt.diff(T2)
        df2du2 = dT2dt.diff(Q2)
        
        return [df1dT2, df1dC2, df2dC2, df2dT2, df2du2]


    def get_matrices_H1(self):
        df1dT1, df1dC1, df2dC1, df2dT1, df2du1 = self.symbolic_diff_H1()
        A = np.array([[df1dT1, df1dC1], [df2dT1, df2dC1]])
        B = np.array([[0], [df2du1]])
        return A, B

    def get_matrices_H2(self):
        df1dT2, df1dC2, df2dC2, df2dT2, df2du = self.symbolic_diff_H2()
        A = np.array([[df1dT2, df1dC2], [df2dT2, df2dC2]])
        B = np.array([[0], [df2du]])
        return A, B

    def evaluate_matrices_H1(self, T, C, u):
        A = copy.deepcopy(self.A1)
        B = copy.deepcopy(self.B1)
        T1 = sp.Symbol('T1')
        C1 = sp.Symbol('C1')
        Q1 = sp.Symbol('Q1')
        A[0][0] = sp.lambdify([T1, C1, Q1], A[0][0], 'numpy')
        a00 = A[0][0](T, C, u)
        A[0][1] = sp.lambdify([T1, C1, Q1], A[0][1], 'numpy')
        a01 = A[0][1](T, C, u)
        A[1][0] = sp.lambdify([T1, C1],  A[1][0], 'numpy')
        a10 = A[1][0](T, C)
        A[1][1] = sp.lambdify([T1, C1], A[1][1], 'numpy')
        a11 = A[1][1](T, C)
        B[1][0] = sp.lambdify([T1, C1], B[1][0], 'numpy')
        b01 = B[1][0](T, C)
        AA = np.array([[a00, a01], [a10, a11]])
        BB = np.array([[0], [b01]])
        return AA, BB
    
    def evaluate_matrices_H2(self, T, C, u):
        A = copy.deepcopy(self.A2)
        B = copy.deepcopy(self.B2)
        T2 = sp.Symbol('T2')
        C2 = sp.Symbol('C2')
        Q2 = sp.Symbol('Q2')
        A[0][0] = sp.lambdify([T2, C2, Q2], A[0][0], 'numpy')
        a00 = A[0][0](T, C, u)
        A[0][1] = sp.lambdify([T2, C2, Q2], A[0][1], 'numpy')
        a01 = A[0][1](T, C, u)
        A[1][0] = sp.lambdify([T2, C2],  A[1][0], 'numpy')
        a10 = A[1][0](T, C)
        A[1][1] = sp.lambdify([T2, C2], A[1][1], 'numpy')
        a11 = A[1][1](T, C)
        B[1][0] = sp.lambdify([T2, C2], B[1][0], 'numpy')
        b01 = B[1][0](T, C)
        AA = np.array([[a00, a01], [a10, a11]])
        BB = np.array([[0], [b01]])
        return AA, BB

    def predict(self, x_k_1, u_k, dt, flag): # prom: x_k = T1[i+1], T2[i+1], C1[i+1], C2[i+1]
        x_k = self.model.simple_step(x_k_1, u_k, dt, flag)  # x_k|k-1: predicted state estimate
        if flag == 'H1':
            F_k = self.evaluate_matrices_H1(x_k[0], x_k[1], u_k)[0]  # Jacobian of the dynamics at the predicted state
        elif flag == 'H2':
            F_k = self.evaluate_matrices_H2(x_k[0], x_k[1], u_k)[0]  # Jacobian of the dynamics at the predicted state
        P_k = np.matmul(np.matmul(F_k, self.P_k_1), np.transpose(F_k)) + self.Q_k  # P_k|k-1: predicted covariance estimate
        return x_k, P_k

    def update(self, x_k, y_k, P_k, H_k):
        y_k = np.array(y_k)
        x_k = np.array(x_k)
        z_k = y_k - x_k  # z_k: innovation
        S_k = self.R_k + np.matmul((np.matmul(H_k, P_k)), np.transpose(H_k))  # S_k: residual covariance
        K_k = np.matmul(np.matmul(P_k, np.transpose(H_k)), np.linalg.inv(S_k)) # K_k: near optimal Kalman gain
        x_kk = x_k + np.dot(K_k, z_k)  # x_k|k: updated state estimate
        P_kk = np.matmul((np.eye(2) - np.matmul(K_k, H_k)), P_k) # P_k|k: updated covariance estimate
        self.P_k_1 = copy.deepcopy(P_kk)
        return x_kk

    def observe(self, y, u, H_k, dt, flag):
        x_k_1 = copy.deepcopy(y)
        x_k, P_k = self.predict(x_k_1, u, dt, flag)
        return self.update(x_k, y, P_k, H_k)
