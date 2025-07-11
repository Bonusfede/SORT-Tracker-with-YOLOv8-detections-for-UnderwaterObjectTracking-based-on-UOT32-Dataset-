import numpy as np
from numpy.linalg import inv

class KalmanFilter():
    def __init__(self,
                 xinit: int = 0,
                 yinit: int = 0,
                 fps: int = 30,
                 std_a: float = 0.001,
                 std_x: float = 0.0045,
                 std_y: float = 0.01,
                 cov: float = 100000) -> None:

        # Matrice di stato
        self.S = np.array([xinit, 0, 0, yinit, 0, 0])
        self.dt = 1 / fps

        # Modello di stato per la velocità (cinematica newtoniana)
        self.F = np.array([[1, self.dt, 0.5 * (self.dt * self.dt), 0, 0, 0],
                           [0, 1, self.dt, 0, 0, 0], [0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 1, self.dt, 0.5 * self.dt * self.dt],
                           [0, 0, 0, 0, 1, self.dt], [0, 0, 0, 0, 0, 1]])

        self.std_a = std_a

        # Noise Q
        self.Q = np.array([
            [
                0.25 * self.dt * self.dt * self.dt * self.dt, 0.5 * self.dt *
                self.dt * self.dt, 0.5 * self.dt * self.dt, 0, 0, 0
            ],
            [
                0.5 * self.dt * self.dt * self.dt, self.dt * self.dt, self.dt,
                0, 0, 0
            ], [0.5 * self.dt * self.dt, self.dt, 1, 0, 0, 0],
            [
                0, 0, 0, 0.25 * self.dt * self.dt * self.dt * self.dt,
                0.5 * self.dt * self.dt * self.dt, 0.5 * self.dt * self.dt
            ],
            [
                0, 0, 0, 0.5 * self.dt * self.dt * self.dt, self.dt * self.dt,
                self.dt
            ], [0, 0, 0, 0.5 * self.dt * self.dt, self.dt, 1]
        ]) * self.std_a * self.std_a

        self.std_x = std_x
        self.std_y = std_y

        # Misurazione Noise
        self.R = np.array([[self.std_x * self.std_x, 0],
                           [0, self.std_y * self.std_y]])

        self.cov = cov

        # Stima incertezza
        self.P = np.array([[self.cov, 0, 0, 0, 0, 0],
                           [0, self.cov, 0, 0, 0, 0],
                           [0, 0, self.cov, 0, 0, 0],
                           [0, 0, 0, self.cov, 0, 0],
                           [0, 0, 0, 0, self.cov, 0],
                           [0, 0, 0, 0, 0, self.cov]])

        # Matrice di osservazine

        self.H = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]])

        self.I = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])

        # Predizione prossimo stato (al tempo t)
        self.S_pred = None
        self.P_pred = None

        # Kalman Gain
        self.K = None

        # S=stato K=gain P=incertezza
        self.S_hist = [self.S]
        self.K_hist = []
        self.P_hist = [self.P]

    def pred_new_state(self):
        self.S_pred = self.F.dot(self.S)

    def pred_next_uncertainity(self):
        self.P_pred = self.F.dot(self.P).dot(self.F.T) + self.Q

    def get_Kalman_gain(self):
        self.K = self.P_pred.dot(self.H.T).dot(
            inv(self.H.dot(self.P_pred).dot(self.H.T) + self.R))
        self.K_hist.append(self.K)

    # Correzioni a S e P
    def state_correction(self, z):
        if z == [None, None]:
            self.S = self.S_pred
        else:
            self.S = self.S_pred + +self.K.dot(z - self.H.dot(self.S_pred))

        self.S_hist.append(self.S)

    def uncertainity_correction(self, z):
        if z != [None, None]:
            self.l1 = self.I - self.K.dot(self.H)
            self.P = self.l1.dot(self.P_pred).dot(self.l1.T) + self.K.dot(
                self.R).dot(self.K.T)
        self.P_hist.append(self.P)


@staticmethod
def cost_fun(a, b):         # Funzione costo per assegnazione del filtro, usa la distanza euclidea
    sm = 0
    for i in range(len(a)):
        sm += (a[i] - b[i])**2
    return sm