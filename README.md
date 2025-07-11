# SORT-Simple-Online-Object-Tracking-for-UnderwaterObjectTracking-based-on-UOT32-Dataset-
Implementation of SORT (Simple-Online-and-Realtime-Object-Tracking) for tracking underwater objects. Utilizza le rilevazioni da un modello di YOLOv8 allenato su un custom dataset (UOT32) e utilizza i filtri di Kalman per determinare la posizione e la velocità al tempo t considerando la posizione e la veloctià al tempo t-1. Utilizza poi l'Hungarian Algorithm per le sovrapposizioni e le indecisioni sulle bounding boxes da disegnare. I filtri di Kalman per semplicità seguono la cinematica newtoniana e la confidence minima per le detection di YOLO di default sono di 0.25 mentre la soglia delle IoU (Intersection-over-Union) è di 0.45.
Nel file utils.py sono disponibili le funzioni ausiliare per la generazione del video in out, per creare un pandas dataframe dalle rilevazioni di YOLOv8 e per salvare in formato Pickle.
Nel file detections.py si utilizza il modello allenato sul dataset UOT32 e di calcolano le detections per poi salvare in un dataframe in formato .pkl
Nel file kalmanFilter.py è presente l'intera classe del filtro di kalman con le sue funzioni ausiliarie
Infine nel file sort_tracker.py avviene il tracking utilizzando i filtri di kalman.


La gif qua sotto è solamente una preview del tracker in azione:
![demo_video (1)](https://github.com/user-attachments/assets/f562a3f8-9d2b-43d8-b9da-4414ef2ec044)

