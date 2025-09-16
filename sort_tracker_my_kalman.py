import pandas as pd
from numpy.linalg import inv
import numpy as np
import math
import cv2
from ultralytics import YOLO
from utils import *         # Import delle mie funzioni ausiliarie
from kalmanFilter import KalmanFilter, cost_fun    # Import della classe del filtro di Kalman


# Conversione del video in una lista di frame
imgs, fps = conv_video_in_frames('video_in/ArmyDiver1.mp4')

# Ottenimento lista di dataframe delle detections
df_list = restore_df_pickle('df_list_army.pkl')

# Implementazione di un dizionario per rendere dinamico il numero di filtri di Kalman con chiave (nome_classe, id) valore -> filtro di kalman
kalman_filters = {}

assig = []
for frame_idx, df in enumerate(df_list):
    detections = []

    # Estrai centro bbox per ogni detection, di tutte le classi
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
            cost = cost_fun([kf.S_hist[-1][0], kf.S_hist[-1][3]], det_coord)
            if cost < min_cost:
                min_cost = cost
                best_id = obj_id

        if best_id is not None:
            matched_detections[(name, best_id)] = (idx, det_coord)
            used_filters.add((name, best_id))
        else:
            # Nuovo oggetto di questa classe
            new_id = max([i for (cls, i) in kalman_filters.keys() if cls == name], default=-1) + 1
            kalman_filters[(name, new_id)] = KalmanFilter(
                fps=fps,
                xinit=det_coord[0],
                yinit=det_coord[1],
                std_x=0.000025,
                std_y=0.0001
            )
            matched_detections[(name, new_id)] = (idx, det_coord)
            used_filters.add((name, new_id))

    for (class_name, obj_id), kf in kalman_filters.items():
        kf.pred_new_state()
        kf.pred_next_uncertainity()
        kf.get_Kalman_gain()
        if (class_name, obj_id) in matched_detections:
            _, coord = matched_detections[(class_name, obj_id)]
            kf.state_correction(coord)
            kf.uncertainity_correction(coord)
        else:
            kf.S_hist.append(kf.S_pred)
            kf.P_hist.append(kf.P_pred)

    for (class_name, obj_id), (idx, _) in matched_detections.items():
        assig.append((frame_idx, idx, obj_id, class_name))

#Salvataggio video in output con il fatto che invece di avere adesso 2 filtri, ne aggiorno ognuno per ogni filtro nel dizionario
out = cv2.VideoWriter('video_out/out_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30,
                      (imgs[0].shape[1], imgs[0].shape[0]))

for i, img in enumerate(imgs):
    tmp_img = img.copy()
    df = df_list[i]

    # Cerchi predizioni Kalman
    for (class_name, obj_id), kf in kalman_filters.items():
        if i < len(kf.S_hist):
            x = math.floor(kf.S_hist[i][0])
            y = math.floor(kf.S_hist[i][3])
            color = ((37 * obj_id) % 255, (79 * obj_id) % 255, (113 * obj_id) % 255)
            tmp_img = cv2.circle(tmp_img, (x, y), radius=3, color=color, thickness=2)

    # Bounding box originali
    for frame_idx, det_idx, obj_id, class_name in assig:
        if frame_idx == i and det_idx in df.index:
            label = f'{class_name}_{obj_id}'
            tmp_img = disegna_predizione(tmp_img, label, df.loc[det_idx], color=(0, 255, 0))

    out.write(tmp_img)


out.release()
