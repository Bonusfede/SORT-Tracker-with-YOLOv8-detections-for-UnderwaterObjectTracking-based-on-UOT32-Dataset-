import pandas as pd
import numpy as np
import math
import cv2
from ultralytics import YOLO
from utils import *  

# Modello CAM (Constant Acceleration Model) quindi accelerazione costante
class OpenCVKalmanFilter:
    def __init__(self, xinit, yinit, fps, std_a=5e-2, std_x=1e-2, std_y=1e-2):
        dt = 1.0 / fps
        self.kf = cv2.KalmanFilter(6, 2)   #x, vx, ax, y, vy, ay

        self.kf.statePre = np.array([[xinit], [0], [0], [yinit], [0], [0]], dtype=np.float32)

        # moto uniformemente accelerato
        self.kf.transitionMatrix = np.array([
            [1, dt, 0.5 * dt**2, 0, 0, 0],          # x (moto uniformemente accelerato)
            [0, 1, dt, 0, 0, 0],                    # vx
            [0, 0, 1, 0, 0, 0],                     # ax (costante)
            [0, 0, 0, 1, dt, 0.5 * dt**2],          # y
            [0, 0, 0, 0, 1, dt],                    # vy
            [0, 0, 0, 0, 0, 1]                      # ay 
        ], dtype=np.float32)

        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0]
        ], dtype=np.float32)

        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * (std_a ** 2)

        self.kf.measurementNoiseCov = np.array([
            [std_x ** 2, 0],
            [0, std_y ** 2]
        ], dtype=np.float32)

        self.kf.errorCovPre = np.eye(6, dtype=np.float32) * 1e5

        self.prediction = self.kf.statePre.copy()
        self.prediction_history = [self.prediction.copy()]

    def predict(self):
        self.prediction = self.kf.predict()
        self.prediction_history.append(self.prediction.copy())
        return self.prediction

    def correct(self, z):
        if z is not None:
            measurement = np.array([[np.float32(z[0])], [np.float32(z[1])]])
            self.kf.correct(measurement)


def cost_fun(a, b):
    return sum((a[i] - b[i])**2 for i in range(len(a)))


imgs, fps = conv_video_in_frames('video_in/WhaleAtBeach2.mp4')
df_list = restore_df_pickle('df_pickles/df_WhaleAtBeach2.pkl')

kalman_filters = {}
assig = []

for frame_idx, df in enumerate(df_list):
    detections = []
    for i in df.index:
        name = df.at[i, 'name']
        x_center = (df.at[i, 'xmin'] + df.at[i, 'xmax']) / 2
        y_center = (df.at[i, 'ymin'] + df.at[i, 'ymax']) / 2
        detections.append(((x_center, y_center), i, name))

    used_filters = set()
    matched_detections = {}

    for det_coord, idx, name in detections:
        min_cost = float('inf')
        best_id = None

        for (class_name, obj_id), kf in kalman_filters.items():
            if class_name != name or (class_name, obj_id) in used_filters:
                continue
            pred = kf.prediction_history[-1]
            cost = cost_fun([pred[0][0], pred[3][0]], det_coord)
            if cost < min_cost:
                min_cost = cost
                best_id = obj_id

        if best_id is not None:
            matched_detections[(name, best_id)] = (idx, det_coord)
            used_filters.add((name, best_id))
        else:
            new_id = max([i for (cls, i) in kalman_filters.keys() if cls == name], default=-1) + 1
            kalman_filters[(name, new_id)] = OpenCVKalmanFilter(
                fps=fps,
                xinit=det_coord[0],
                yinit=det_coord[1],
                std_x=1e-2,
                std_y=1e-2
            )
            matched_detections[(name, new_id)] = (idx, det_coord)
            used_filters.add((name, new_id))

    for (class_name, obj_id), kf in kalman_filters.items():
        kf.predict()
        if (class_name, obj_id) in matched_detections:
            _, coord = matched_detections[(class_name, obj_id)]
            kf.correct(coord)

    for (class_name, obj_id), (idx, _) in matched_detections.items():
        assig.append((frame_idx, idx, obj_id, class_name))



out = cv2.VideoWriter('video_out/whale_at_beach2_a_cost.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30,
                      (imgs[0].shape[1], imgs[0].shape[0]))

for i, img in enumerate(imgs):
    tmp_img = img.copy()
    df = df_list[i]

    for (class_name, obj_id), kf in kalman_filters.items():
        if i < len(kf.prediction_history):
            x = int(kf.prediction_history[i][0])
            y = int(kf.prediction_history[i][3])
            color = (0, 0, 255)
            tmp_img = cv2.circle(tmp_img, (x, y), radius=1, color=color, thickness=3)

    for frame_idx, det_idx, obj_id, class_name in assig:
        if frame_idx == i and det_idx in df.index:
            label = f'{class_name}_{obj_id}'
            tmp_img = disegna_predizione(tmp_img, label, df.loc[det_idx], color=(0, 255, 0))

    out.write(tmp_img)

out.release()
