from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter
import numpy as np


class UKF:
    def __init__(self, n, m, dt):
        self.n = n
        self.m = m
        self.dt = dt
        self.x_prev = np.zeros(n)
        self.p_prev = 0.001 * np.eye(n)

    def statespace(self, x_prev, dt):
        x_pred = np.exp(x_prev)
        return x_pred

    def measurement(self, x_pred):
        z = np.cos(x_pred)
        return z

    def Estimation(self, z):
        fx = self.statespace
        hx = self.measurement

        points = MerweScaledSigmaPoints(self.n, alpha=.1, beta=2., kappa=-1)
        ukf = UnscentedKalmanFilter(dim_x=self.n, dim_z=self.m, dt=self.dt, fx=fx, hx=hx, points=points)

        ukf.x = self.x_prev
        ukf.P = self.p_prev
        ukf.R = (0.01 ** 2) * np.eye(self.m)
        ukf.Q = (0.01 ** 2) * np.eye(self.m)

        ukf.predict()
        ukf.update(z)

        self.x_prev = ukf.x
        self.p_prev = ukf.P
        return ukf.x
