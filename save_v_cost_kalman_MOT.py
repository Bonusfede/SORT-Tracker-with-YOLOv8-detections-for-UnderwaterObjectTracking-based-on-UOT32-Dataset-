import pandas as pd
import numpy as np
import math
import cv2
from ultralytics import YOLO
from utils import * 
import os

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

video = "C:/Users/fedem/Desktop/Progetto/archive"  # dove sono le 32 cartelle
nome_tracker = "KalmanTracker_v_cost"
MOT_out = f"C:/Users/fedem/Desktop/TrackEval-master/TrackEval-master/data/CustomDataset/trackers/{nome_tracker}"
os.makedirs(MOT_out, exist_ok=True)

# Funzione per esportare i risultati in formato MOT_challenge
def export_results_trackeval(df_list, assig, nome_tracker, seq_name, save_dir=MOT_out):
    save_path = os.path.join(save_dir, nome_tracker)
    os.makedirs(save_path, exist_ok=True)
    output_file = os.path.join(save_path, f"{seq_name}.txt")

    lines = []
    # Prende dal dataframe pkl i vari campi delle detections di yolo
    for frame_idx, det_idx, obj_id, class_name in assig:
        det = df_list[frame_idx].loc[det_idx]
        frame = frame_idx + 1
        x = det['xmin']
        y = det['ymin']
        w = det['xmax'] - det['xmin']
        h = det['ymax'] - det['ymin']
        conf = det['confidence']
        lines.append(f"{frame}, {obj_id}, {x}, {y}, {w}, {h}, {conf}, -1, -1, -1")

    with open(output_file, "w") as f:
        f.write("\n".join(lines))

    print(f"File salvato: {output_file}")

for cartella in os.listdir(video):
    dir_cartella = os.path.join(video, cartella)
    if not os.path.isdir(dir_cartella):
        continue

    # Rileva video e pkl
    video_file = os.path.join(video, cartella, f"{cartella}.mp4")
    for f in os.listdir(dir_cartella):
        if f.endswith((".mp4", ".avi")):
            video_file = os.path.join(dir_cartella, f)
            break

    pkl_file = os.path.join("df_pickles", f"df_{cartella}.pkl")
    if not os.path.isfile(video_file) or not os.path.isfile(pkl_file):
        continue

    print(f"Tracciamento in corso: {cartella}")
    imgs, fps = conv_video_in_frames(video_file)
    df_list = restore_df_pickle(pkl_file)
    total_frames = len(imgs)

    kalman_filters = {}
    assig = []
    global_id_counter = 0
    id_mapping = {}  # (class_name, local_id) -> global_id

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

        # Costruzione assegnazioni con ID globali e controllo duplicati per frame
        ids_in_frame = set()
        for (class_name, local_id), (idx, _) in matched_detections.items():
            key = (class_name, local_id)
            if key not in id_mapping:
                id_mapping[key] = global_id_counter
                global_id_counter += 1
            global_id = id_mapping[key]

            if global_id in ids_in_frame:
                continue
            ids_in_frame.add(global_id)

            assig.append((frame_idx, idx, global_id, class_name))
            
        for (class_name, obj_id), (idx, _) in matched_detections.items():
            assig.append((frame_idx, idx, obj_id, class_name))
    # Salva risultati in formato MOT_challenge
    export_results_trackeval(df_list, assig, nome_tracker, seq_name=cartella)

print("Tutte le sequenze processate!")
