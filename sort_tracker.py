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

# Otteniamo la lista di dataframe delle detections
df_list = restore_df_pickle('df_list_army.pkl')


# Costruzione della classe del filtro di Kalman (per ora 2 oggetti) # TODO: Cercare di assegnare un filtro di Kalman per object_id rilevato
# k_filter = [
#     KalmanFilter(fps=fps, xinit=60, yinit=150,
#                  std_x=0.000025, std_y=0.0001),
#     KalmanFilter(fps=fps, xinit=620, yinit=150,
#                  std_x=0.000025, std_y=0.0001)
# ]

# Implementazione di un dizionario per rendere dinamico il numero di filtri di Kalman con chiave (nome_classe, id) valore -> filtro di kalman
kalman_filters = {}

assig = []
# Vecchio ciclo per aggionrare i 2 filtri di kalman
# for df in df_list:
#     df = df.loc[df['name'] == 'Diver']
#     x_cen, y_cen = [None, None], [None, None]

#     for i in df.index.values:
#         coord = [(df.at[i, 'xmin'] + df.at[i, 'xmax']) / 2,
#                  (df.at[i, 'ymin'] + df.at[i, 'ymax']) / 2]

#         if cost_fun([
#                 k_filter[0].S_hist[-1][0], k_filter[0].S_hist[-1][3]
#         ], coord) < cost_fun(
#             [k_filter[1].S_hist[-1][0], k_filter[1].S_hist[-1][3]],
#                 coord) and x_cen[0] == None and y_cen[0] == None:
#             x_cen[0], y_cen[0] = coord[0], coord[1]
#             assig.append(0)
#         else:
#             x_cen[1], y_cen[1] = coord[0], coord[1]
#             assig.append(1)

#     for i in range(2):
#         k_filter[i].pred_new_state()
#         k_filter[i].pred_next_uncertainity()
#         k_filter[i].get_Kalman_gain()
#         k_filter[i].state_correction([x_cen[i], y_cen[i]])
#         k_filter[i].uncertainity_correction([x_cen[i], y_cen[i]])

# Nuovo ciclo per aggiornare ogni singolo filtro di kalman presente nel dizionario
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

# Salvataggio video output
# ind = 0
# out = cv2.VideoWriter('video_out/demo_video.avi', cv2.VideoWriter_fourcc(*'mp4v'), 30, (imgs[0].shape[1], imgs[0].shape[0]))

# for i in range(len(imgs)):
#     tmp_img = imgs[i]
#     df = df_list[i].loc[df_list[i]['name'] == 'Diver']

#     tmp_img = cv2.circle(tmp_img, (math.floor(k_filter[0].S_hist[i][0]), math.floor(k_filter[0].S_hist[i][3])), radius=1, color=(255, 0, 0), thickness=3)

#     tmp_img = cv2.circle(tmp_img, (math.floor(k_filter[1].S_hist[i][0]), math.floor(k_filter[1].S_hist[i][3])), radius=1, color=(0, 0, 255), thickness=3)

#     for j in df.index.values:
#         if assig[ind] == 0:
#             tmp_img = disegna_predizione(tmp_img,
#                                       'Diver1', df.loc[j], color=(0, 255, 0))
#         else:
#             tmp_img = disegna_predizione(tmp_img,
#                                       'Diver2', df.loc[j], color=(0, 0, 255))
#         ind += 1

#     out.write(tmp_img)

# out.release()


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

