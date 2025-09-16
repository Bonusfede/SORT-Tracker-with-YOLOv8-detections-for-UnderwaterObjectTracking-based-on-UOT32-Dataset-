import os
from ultralytics import YOLO

# === Impostazioni ===
input_base = "C:/Users/fedem/Desktop/Progetto/archive"  # Directory con le 32 cartelle dei video
output_base = "C:/Users/fedem/Desktop/TrackEval-master/TrackEval-master/data/CustomDataset/trackers"
model_path = "best.pt"  # Puoi anche usare 'yolov8n.pt', 'yolov8m.pt', ecc.

# === Trackers da usare ===
trackers = {
    "BoTSORT": "botsort.yaml",
    "ByteTrack": "bytetrack.yaml"
}

# === Carica il modello una volta sola ===
model = YOLO(model_path)

# === Loop su tutte le cartelle del dataset ===
for folder_name in os.listdir(input_base):
    folder_path = os.path.join(input_base, folder_name)
    if not os.path.isdir(folder_path):
        continue

    # Trova il file video nella cartella
    video_file = None
    for f in os.listdir(folder_path):
        if f.endswith((".mp4", ".avi")):
            video_file = os.path.join(folder_path, f)
            break
    if video_file is None:
        print(f"⚠️ Nessun video trovato in {folder_name}, salto.")
        continue

    # === Tracking con entrambi i tracker ===
    for tracker_name, tracker_cfg in trackers.items():
        print(f"🚀 Tracking {folder_name} con {tracker_name}")

        results = model.track(
            source=video_file,
            persist=True,
            tracker=tracker_cfg,
            stream=False,
            save=False,
            verbose=False
        )

        # === Parsing risultati
        all_tracks = []
        for i, result in enumerate(results):
            ids = result.boxes.id
            if ids is None:
                continue
            for box, track_id, conf in zip(result.boxes.xyxy, result.boxes.id, result.boxes.conf):
                x1, y1, x2, y2 = box.tolist()
                w = x2 - x1
                h = y2 - y1
                frame = i + 1
                all_tracks.append(f"{frame}, {int(track_id)}, {x1:.2f}, {y1:.2f}, {w:.2f}, {h:.2f}, {conf:.3f}, -1, -1, -1")

        # === Salva risultati nel formato TrackEval
        tracker_out_dir = os.path.join(output_base, tracker_name)
        os.makedirs(tracker_out_dir, exist_ok=True)
        output_file = os.path.join(tracker_out_dir, f"{folder_name}.txt")
        with open(output_file, "w") as f:
            f.write("\n".join(all_tracks))
        print(f"✅ Salvato {output_file} con {len(all_tracks)} righe")

print("🎉 Tutte le sequenze processate con BoT-SORT e ByteTrack!")
