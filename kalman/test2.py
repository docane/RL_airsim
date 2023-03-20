import numpy as np


class KalmanFilter:
    def __init__(self, A, B, H, Q, R, P):
        self.A = A
        self.B = B
        self.H = H
        self.Q = Q
        self.R = R
        self.P = P
        self.state = np.zeros((A.shape[0], 1))

    def predict(self, control_input):
        self.state = np.dot(self.A, self.state) + np.dot(self.B, control_input)
        self.P = np.dot(self.A, np.dot(self.P, self.A.T)) + self.Q

    def update(self, measurement):
        innovation = measurement - np.dot(self.H, self.state)
        innovation_covariance = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        kalman_gain = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(innovation_covariance)))
        self.state = self.state + np.dot(kalman_gain, innovation)
        self.P = self.P - np.dot(kalman_gain, np.dot(self.H, self.P))


g = 9.8
m_c = 1.0
m_p = 0.1
l = 1.0
delta_t = 0.02

A = np.array([[1, delta_t],
              [0, 1]])
B = np.array([0, g / m_c]).reshape(2, 1)
H = np.array([1, 0]).reshape(1, 2)
Q = np.array([[1, 0],
              [0, 1]])
R = np.array([[0.01]])
P = np.array([[0.1, 0.1],
              [0.1, 0.1]])

x = np.array([[0],
              [0]])

kf = KalmanFilter(A, B, H, Q, R, P)

for i in range(100):
    u = -1 if x[0] < 0 else 1

    A[0, 1] = delta_t

    x_dot = np.dot(A, x) + np.dot(B, u)
    x = x + x_dot * delta_t

    kf.predict(u)
    y = x[0]
    kf.update(y)

    print(f"Time: {i * delta_t:.2f} s")
    print(f"Cart position: {x[0, 0]:.2f} m")
    print(f"Cart velocity: {x[1, 0]:.2f} m/s")
    print(f"Pole angle: {y[0]:.2f} rad")
    print(f"Pole angular velocity: {x_dot[1, 0]:.2f} rad/s")
    print("\n")
    print(kf.P)

