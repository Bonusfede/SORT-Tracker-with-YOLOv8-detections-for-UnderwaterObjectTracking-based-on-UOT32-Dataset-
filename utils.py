import pandas as pd
import numpy as np
import cv2
import pickle



def disegna_predizione(img: np.ndarray,
                    classe: str,
                    df: pd.core.series.Series,
                    color: tuple = (255, 0, 0)):
    # disegna rettangolo per le pred
    cv2.rectangle(img, (int(df.xmin), int(df.ymin)), (int(df.xmax), int(df.ymax)), color, 2)

    # testo rettangolo
    cv2.putText(img, classe + " " + str(round(df.confidence, 2)),
                (int(df.xmin) - 10, int(df.ymin) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img
    


def conv_video_in_frames(path:str):
    cap = cv2.VideoCapture(path)        # cattura video
    fps = cap.get(cv2.CAP_PROP_FPS)     # framerate del video

    img = []
    while cap.isOpened():
        ret, frame = cap.read()          # legge frame per frame
        if ret == True:
            img.append(frame)                # aggiunge frame alla lista
        else: 
            break

    cap.release()                      
    return img, fps                    # ritorna la lista di frame e il framerate





def detect_get_pandas_df(model, images): 
    # ottiene le rilevazioni da YOLOv8 e le mette in una lista di dataframe
    results = model.predict(images, conf=0.25, iou=0.45, device='cuda')

    df_list = []

    for result in results:
        boxes = result.boxes
        if boxes is not None:
            xyxy = boxes.xyxy.cpu().numpy()        # [xmin, ymin, xmax, ymax, conf, class]
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)

            names = [model.names[c] for c in cls]  # class names

            df = pd.DataFrame(xyxy, columns=['xmin', 'ymin', 'xmax', 'ymax'])
            df['confidence'] = conf
            df['class'] = cls
            df['name'] = names
            df_list.append(df)
        else:
            df_list.append(pd.DataFrame())  # Nessun oggetto trovato

    return df_list





def save_df_to_pickle(df_list, path): # serve per salvare il dataframe al video associato
    with open(path, 'wb') as f:     # salva il file in un formato pkl per poi essere riaperto dopo
	    pickle.dump(df_list, f)





def restore_df_pickle(path):  # utile per ripristinare un dataframe e salvarlo in una nuova variabile senza fare nuovamente le predizioni
    df_ret = []
    with open(path, 'rb') as f:
        df_ret = pickle.load(f)

    return df_ret