import os
from ultralytics import YOLO
from utils import conv_video_in_frames, detect_get_pandas_df, save_df_to_pickle

video = "C:/Users/fedem/Desktop/Progetto/archive"  # cartelle con i video
df_out = "df_pickles"
os.makedirs(df_out, exist_ok=True)

model = YOLO('best.pt')

for folder_name in os.listdir(video):
    folder_path = os.path.join(video, folder_name)
    if not os.path.isdir(folder_path):
        continue

    #Trova il file video nella cartella
    video_file = None
    for f in os.listdir(folder_path):
        if f.endswith((".mp4", ".avi", ".mov")):
            video_file = os.path.join(folder_path, f)
            break

 
    if video_file is None:
        continue

    df_list = detect_get_pandas_df(model, video_file)
    pickle_path = os.path.join(df_out, f"df_{folder_name}.pkl")
    save_df_to_pickle(df_list, pickle_path)
    print(f"Salvato pickle: {pickle_path}")
