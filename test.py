import pandas as pd
from numpy.linalg import inv
import numpy as np
import math
import cv2
import torch
from ultralytics import YOLO

from matplotlib import pyplot as plt
import hashlib

model = YOLO('best.pt', verbose=False)
device = torch.device('cuda')
model = model.to(device)

def draw_prediction(img: np.ndarray,
                    class_name: str,
                    df: pd.core.series.Series,
                    color: tuple = (255, 0, 0)):
    '''
    Function to draw prediction around the bounding box identified by the YOLO
    The Function also displays the confidence score top of the bounding box 
    '''

    cv2.rectangle(img, (int(df.xmin), int(df.ymin)),
                  (int(df.xmax), int(df.ymax)), color, 2)
    cv2.putText(img, class_name + " " + str(round(df.confidence, 2)),
                (int(df.xmin) - 10, int(df.ymin) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img

def convert_video_to_frame(path: str):
    '''
    The function take input as video file and returns a list of images for every video
    '''

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    img = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            img.append(frame)
        else:
            break

    cap.release()
    return img, fps


# Converting Video to image frame by frame for a single and multiple ball

img_multi, fps_multi = convert_video_to_frame('video_in/ArmyDiver1.mp4')


result_multi = model(img_multi)
#results_multi = model(img_multi)

#df_sin = results_sin.pandas().xyxy
#df_multi = results_multi.pandas().xyxy

#print(df_sin)

results_sin = model.predict(img_multi, conf=0.25, iou=0.45, device='cuda')

df_multi = []

for result in results_sin:
    boxes = result.boxes
    if boxes is not None:
        xyxy = boxes.xyxy.cpu().numpy()        # [xmin, ymin, xmax, ymax, conf, class]
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)

        names = [model.names[c] for c in cls]  # class names

        df = pd.DataFrame(xyxy, columns=['xmin', 'ymin', 'xmax', 'ymax'])
        df['confidence'] = conf
        df['class'] = cls
        df['name'] = names
        df_multi.append(df)
    else:
        df_multi.append(pd.DataFrame())  # Nessun oggetto trovato

print(df_multi[0])


class KalmanFilter():
    def __init__(self,
                 xinit: int = 0,
                 yinit: int = 0,
                 fps: int = 30,
                 std_a: float = 0.001,
                 std_x: float = 0.0045,
                 std_y: float = 0.01,
                 cov: float = 100000) -> None:

        # State Matrix
        self.S = np.array([xinit, 0, 0, yinit, 0, 0])
        self.dt = 1 / fps

        # State Transition Model
        # Here, we assume that the model follow Newtonian Kinematics
        self.F = np.array([[1, self.dt, 0.5 * (self.dt * self.dt), 0, 0, 0],
                           [0, 1, self.dt, 0, 0, 0], [0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 1, self.dt, 0.5 * self.dt * self.dt],
                           [0, 0, 0, 0, 1, self.dt], [0, 0, 0, 0, 0, 1]])

        self.std_a = std_a

        # Process Noise
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

        # Measurement Noise
        self.R = np.array([[self.std_x * self.std_x, 0],
                           [0, self.std_y * self.std_y]])

        self.cov = cov

        # Estimate Uncertainity
        self.P = np.array([[self.cov, 0, 0, 0, 0, 0],
                           [0, self.cov, 0, 0, 0, 0],
                           [0, 0, self.cov, 0, 0, 0],
                           [0, 0, 0, self.cov, 0, 0],
                           [0, 0, 0, 0, self.cov, 0],
                           [0, 0, 0, 0, 0, self.cov]])

        # Observation Matrix
        # Here, we are observing X & Y (0th index and 3rd Index)
        self.H = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]])

        self.I = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])

        # Predicting the next state and estimate uncertainity
        self.S_pred = None
        self.P_pred = None

        # Kalman Gain
        self.K = None

        # Storing all the State, Kalman Gain and Estimate Uncertainity
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


filter_multi = [
    KalmanFilter(fps=fps_multi, xinit=60, yinit=150,
                 std_x=0.000025, std_y=0.0001),
    KalmanFilter(fps=fps_multi, xinit=620, yinit=150,
                 std_x=0.000025, std_y=0.0001)
]


def cost_fun(a, b):
    '''
    Cost function for filter Assignment
    Uses euclidean distance for choosing the filter
    '''

    sm = 0
    for i in range(len(a)):
        sm += (a[i] - b[i])**2
    return sm


assig = []


for df in df_multi:
    df = df.loc[df['name'] == 'Diver']
    x_cen, y_cen = [None, None], [None, None]

    for i in df.index.values:
        coord = [(df.at[i, 'xmin'] + df.at[i, 'xmax']) / 2,
                 (df.at[i, 'ymin'] + df.at[i, 'ymax']) / 2]

        if cost_fun([
                filter_multi[0].S_hist[-1][0], filter_multi[0].S_hist[-1][3]
        ], coord) < cost_fun(
            [filter_multi[1].S_hist[-1][0], filter_multi[1].S_hist[-1][3]],
                coord) and x_cen[0] == None and y_cen[0] == None:
            x_cen[0], y_cen[0] = coord[0], coord[1]
            assig.append(0)
        else:
            x_cen[1], y_cen[1] = coord[0], coord[1]
            assig.append(1)

    for i in range(2):
        filter_multi[i].pred_new_state()
        filter_multi[i].pred_next_uncertainity()
        filter_multi[i].get_Kalman_gain()
        filter_multi[i].state_correction([x_cen[i], y_cen[i]])
        filter_multi[i].uncertainity_correction([x_cen[i], y_cen[i]])


ind = 0
out = cv2.VideoWriter('dfsf.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30,
                      (img_multi[0].shape[1], img_multi[0].shape[0]))

for i in range(len(img_multi)):
    tmp_img = img_multi[i]
    df = df_multi[i].loc[df_multi[i]['name'] == 'Diver']

    tmp_img = cv2.circle(tmp_img, (math.floor(filter_multi[0].S_hist[i][0]),
                                   math.floor(filter_multi[0].S_hist[i][3])),
                         radius=1,
                         color=(255, 0, 0),
                         thickness=3)
    tmp_img = cv2.circle(tmp_img, (math.floor(filter_multi[1].S_hist[i][0]),
                                   math.floor(filter_multi[1].S_hist[i][3])),
                         radius=1,
                         color=(0, 0, 255),
                         thickness=3)

    for j in df.index.values:
        if assig[ind] == 0:
            tmp_img = draw_prediction(tmp_img,
                                      'Ball 1',
                                      df.loc[j],
                                      color=(255, 0, 0))
        else:
            tmp_img = draw_prediction(tmp_img,
                                      'Ball 2',
                                      df.loc[j],
                                      color=(0, 0, 255))
        ind += 1

    out.write(tmp_img)

out.release()