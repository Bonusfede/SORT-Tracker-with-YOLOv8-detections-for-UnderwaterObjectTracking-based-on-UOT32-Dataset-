import os
import cv2

dataset_video = "C:/Users/fedem/Desktop/Progetto/archive"  # ← percorso delle 32 cartelle
mot_out = "C:/Users/fedem/Desktop/TrackEval-master/TrackEval-master/data/CustomDataset"  # destinazione struttura TrackEval

# 
for cartella in os.listdir(dataset_video):
    dir_cartella = os.path.join(dataset_video, cartella)
    gt_file = os.path.join(dir_cartella, "groundtruth_rect.txt")
    video_file = None

    if not os.path.isdir(dir_cartella) or not os.path.exists(gt_file):
        continue

    # Trova il video .mp4 di UOT32
    for f in os.listdir(dir_cartella):
        if f.endswith((".mp4")):
            video_file = os.path.join(dir_cartella, f)
            break

    # Prendi info dal video (risoluzione, fps)
    if video_file:
        cap = cv2.VideoCapture(video_file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    else:
        with open(gt_file, "r") as f:
            length = len(f.readlines())

    # Crea directory di output
    seq_dir = os.path.join(mot_out, cartella)
    os.makedirs(os.path.join(seq_dir, "gt"), exist_ok=True)

    # Converte groundtruth.txt in gt.txt (formato MOT)
    with open(gt_file, "r") as f:
        lines = f.readlines()

    gt_lines = []
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) != 4:
            continue
        x, y, w, h = map(float, parts)
        frame = i + 1  # MOT è 1-based
        track_id = 1
        conf = 1
        gt_lines.append(f"{frame}, {track_id}, {x:.2f}, {y:.2f}, {w:.2f}, {h:.2f}, {conf}, -1, -1, -1")

    with open(os.path.join(seq_dir, "gt", "gt.txt"), "w") as f_out:
        f_out.write("\n".join(gt_lines))

    # Scrivi seqinfo.ini
    seqinfo = f"""[Sequence]
name={cartella}
imDir=img1
frameRate={int(fps)}
seqLength={length}
imWidth={width}
imHeight={height}
"""
    with open(os.path.join(seq_dir, "seqinfo.ini"), "w") as f:
        f.write(seqinfo)

print("Dataset MOTChallenge per TrackEval generato con successo.")
