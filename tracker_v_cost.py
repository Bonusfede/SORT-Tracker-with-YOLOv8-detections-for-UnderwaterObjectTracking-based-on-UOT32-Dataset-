import pandas as pd
import numpy as np
import math
import cv2
from ultralytics import YOLO
from utils import * 

# Modello CVM (Constant velocity model) velocità costante
class OpenCVKalmanFilter:
    # valori std_a per accelerazioni improvvise, std_x e std_y per il rumore in immagini subacque
    def __init__(self, xinit, yinit, total_frames, fps=30, std_a=5e-2, std_x=1e-2, std_y=1e-2):    
        dt = 1.0 / fps
        self.kf = cv2.KalmanFilter(4, 2)

        self.kf.statePre = np.array([[xinit], [0], [yinit], [0]], dtype=np.float32)

        self.kf.transitionMatrix = np.array([
            [1, dt, 0,  0],   # x_new = x + vx * dt
            [0,  1, 0,  0],   # vx costante
            [0,  0, 1, dt],   # y_new = y + vy * dt
            [0,  0, 0,  1]    # vy costante
        ], dtype=np.float32)

        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0]
        ], dtype=np.float32)

        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * (std_a ** 2)

        self.kf.measurementNoiseCov = np.array([
            [std_x ** 2, 0],
            [0, std_y ** 2]
        ], dtype=np.float32)

        self.kf.errorCovPre = np.eye(4, dtype=np.float32) * 1e5

        self.prediction = self.kf.statePre.copy()
        self.prediction_history = [None] * total_frames  # lunghezza fissa
        self.missed_frames = 0


    def predict(self, frame_idx):
        self.prediction = self.kf.predict()
        self.prediction_history[frame_idx] = self.prediction.copy()
        return self.prediction

    def correct(self, z, w=None, h=None):
        if z is not None:
            measurement = np.array([[np.float32(z[0])], [np.float32(z[1])]])
            self.kf.correct(measurement)
            if w is not None and h is not None:
                self.width = w
                self.height = h

def cost_fun(a, b):
    return sum((a[i] - b[i])**2 for i in range(len(a)))


imgs, fps = conv_video_in_frames('video_in/ArmyDiver3.mp4')
df_list = restore_df_pickle('df_pickles/df_ArmyDiver3.pkl')
total_frames = len(imgs)

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
            pred = kf.prediction_history[frame_idx - 1] if frame_idx > 0 else kf.prediction
            if pred is None:
                continue
            cost = cost_fun([pred[0][0], pred[2][0]], det_coord)
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
                total_frames=total_frames,
                std_x=1e-2,
                std_y=1e-2
            )
            matched_detections[(name, new_id)] = (idx, det_coord)
            used_filters.add((name, new_id))

    for (class_name, obj_id), kf in kalman_filters.items():
        kf.predict(frame_idx)
        if (class_name, obj_id) in matched_detections:
            idx, coord = matched_detections[(class_name, obj_id)]
            box = df.loc[idx]
            width = int(box["xmax"] - box["xmin"])
            height = int(box["ymax"] - box["ymin"])
            kf.correct(coord, width, height)

    for (class_name, obj_id), (idx, _) in matched_detections.items():
        assig.append((frame_idx, idx, obj_id, class_name))

out = cv2.VideoWriter('video_out/Army3_vcost.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30,
                      (imgs[0].shape[1], imgs[0].shape[0]))

for i, img in enumerate(imgs):
    tmp_img = img.copy()
    df = df_list[i]

    for (class_name, obj_id), kf in kalman_filters.items():
        pred = kf.prediction_history[i]
        if pred is not None:
            x = int(pred[0])
            y = int(pred[2])
            color = (0, 0, 255)
            tmp_img = cv2.circle(tmp_img, (x, y), radius=1, color=color, thickness=3)

    for frame_idx, det_idx, obj_id, class_name in assig:
        if frame_idx == i and det_idx in df.index:
            label = f'{class_name}_{obj_id}'
            tmp_img = disegna_predizione(tmp_img, label, df.loc[det_idx], color=(0, 255, 0))

    out.write(tmp_img)

out.release()
