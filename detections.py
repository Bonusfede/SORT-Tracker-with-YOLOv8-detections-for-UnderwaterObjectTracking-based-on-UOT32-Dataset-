from ultralytics import YOLO
from utils import * 
import torch

# Caricamento del rilevatore (YOLOv8 in questo caso)
model = YOLO('best.pt')     # modello già allenato 

# Conversione del video in una lista di frame
images, fpsses = conv_video_in_frames('video_in/ArmyDiver1.mp4')

# Predizioni e salvataggio del dataframe sul video
df_list = detect_get_pandas_df(model, images)

save_df_to_pickle(df_list, 'df_list_army.pkl')  