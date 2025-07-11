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
k_filter = [
    KalmanFilter(fps=fps, xinit=60, yinit=150,
                 std_x=0.000025, std_y=0.0001),
    KalmanFilter(fps=fps, xinit=620, yinit=150,
                 std_x=0.000025, std_y=0.0001)
]


assig = []

for df in df_list:
    df = df.loc[df['name'] == 'Diver']
    x_cen, y_cen = [None, None], [None, None]

    for i in df.index.values:
        coord = [(df.at[i, 'xmin'] + df.at[i, 'xmax']) / 2,
                 (df.at[i, 'ymin'] + df.at[i, 'ymax']) / 2]

        if cost_fun([
                k_filter[0].S_hist[-1][0], k_filter[0].S_hist[-1][3]
        ], coord) < cost_fun(
            [k_filter[1].S_hist[-1][0], k_filter[1].S_hist[-1][3]],
                coord) and x_cen[0] == None and y_cen[0] == None:
            x_cen[0], y_cen[0] = coord[0], coord[1]
            assig.append(0)
        else:
            x_cen[1], y_cen[1] = coord[0], coord[1]
            assig.append(1)

    for i in range(2):
        k_filter[i].pred_new_state()
        k_filter[i].pred_next_uncertainity()
        k_filter[i].get_Kalman_gain()
        k_filter[i].state_correction([x_cen[i], y_cen[i]])
        k_filter[i].uncertainity_correction([x_cen[i], y_cen[i]])


# Salvataggio video output
ind = 0
out = cv2.VideoWriter('video_out/armydiver.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (imgs[0].shape[1], imgs[0].shape[0]))

for i in range(len(imgs)):
    tmp_img = imgs[i]
    df = df_list[i].loc[df_list[i]['name'] == 'Diver']

    tmp_img = cv2.circle(tmp_img, (math.floor(k_filter[0].S_hist[i][0]), math.floor(k_filter[0].S_hist[i][3])), radius=1, color=(255, 0, 0), thickness=3)

    tmp_img = cv2.circle(tmp_img, (math.floor(k_filter[1].S_hist[i][0]), math.floor(k_filter[1].S_hist[i][3])), radius=1, color=(0, 0, 255), thickness=3)

    for j in df.index.values:
        if assig[ind] == 0:
            tmp_img = disegna_predizione(tmp_img,
                                      'Diver1', df.loc[j], color=(0, 255, 0))
        else:
            tmp_img = disegna_predizione(tmp_img,
                                      'Diver2', df.loc[j], color=(0, 0, 255))
        ind += 1

    out.write(tmp_img)

out.release()