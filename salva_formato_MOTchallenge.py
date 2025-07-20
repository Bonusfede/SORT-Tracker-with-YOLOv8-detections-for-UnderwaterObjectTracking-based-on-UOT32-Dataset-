import os
import cv2
import math
import numpy as np
from kalmanFilter import KalmanFilter, cost_fun
from utils import conv_video_in_frames, restore_df_pickle, disegna_predizione

video = "C:/Users/fedem/Desktop/Progetto/archive"  # dove sono le 32 cartelle
nome_tracker = "KalmanTracker"
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
                cost = cost_fun([kf.S_hist[-1][0], kf.S_hist[-1][3]], det_coord)
                if cost < min_cost:
                    min_cost = cost
                    best_id = obj_id

            if best_id is not None:
                matched_detections[(name, best_id)] = (idx, det_coord)
                used_filters.add((name, best_id))
            else:
                new_local_id = max([i for (cls, i) in kalman_filters.keys() if cls == name], default=-1) + 1
                kalman_filters[(name, new_local_id)] = KalmanFilter(
                    fps=fps, xinit=det_coord[0], yinit=det_coord[1],
                    std_x=0.000025, std_y=0.0001
                )
                matched_detections[(name, new_local_id)] = (idx, det_coord)
                used_filters.add((name, new_local_id))

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
