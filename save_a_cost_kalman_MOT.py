import pandas as pd
import numpy as np
import math
import cv2
from ultralytics import YOLO
from utils import * 
import os 

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

video = "C:/Users/fedem/Desktop/Progetto/archive"  # dove sono le 32 cartelle
nome_tracker = "KalmanTracker_A_cost"
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
# imgs, fps = conv_video_in_frames('video_in/FishFollowing.mp4')
# df_list = restore_df_pickle('df_pickles/df_FishFollowing.pkl')
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



    # Salva risultati in formato MOT_challenge
    export_results_trackeval(df_list, assig, nome_tracker, seq_name=cartella)

print("Tutte le sequenze processate!")

