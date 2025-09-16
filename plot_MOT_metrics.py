import pandas as pd
import matplotlib.pyplot as plt

# === Dati estratti manualmente dalla tabella "COMBINED" per ogni metrica ===
data = {
    "Sequence": ["COMBINED"],
    "HOTA": [78.092],
    "MOTA": [87.969],
    "IDF1": [92.859]
}

# Sezioni con valori per tutte le sequenze
sequences = [
    "ArmyDiver1", "ArmyDiver2", "ArmyDiver3", "Ballena", "BlueFish1", "BlueFish2", "BoySwimming",
    "CenoteAngelita", "DeepSeaFish", "Dolphin1", "Dolphin2", "FishFollowing", "Fisherman",
    "GarryFish", "HoverFish1", "HoverFish2", "JerkbaitBites", "MonsterCreature1",
    "MonsterCreature2", "Octopus1", "Octopus2", "PinkFish", "SeaDiver", "SeaDragon",
    "SeaTurtle1", "SeaTurtle2", "SeaTurtle3", "Steinlager", "WhaleAtBeach1", "WhaleAtBeach2",
    "WhaleDiving", "WhiteShark"
]

hota = [
    50.146, 53.193, 92.58, 77.352, 72.969, 71.298, 88.903, 89.322, 85.962, 67.811, 79.754,
    57.183, 86.623, 63.081, 80.366, 89.525, 71.293, 91.262, 86.837, 60.657, 79.845, 40.52,
    85.298, 77.388, 96.083, 90.536, 85.555, 82.133, 80.465, 90.578, 76.572, 82.098
]

mota = [
    10.965, 7.4286, 99.893, 95.275, 93.412, 84.486, 100, 98.645, 99.916, 88.312, 96.154,
    77.605, 98.87, 73.517, 94.056, 100, 82.752, 99.878, 91.667, 75.391, 98.665, 41.966,
    99.756, 84.216, 99.771, 100, 98.747, 100, 98.891, 98.738, 84.167, 91.103
]

idf1 = [
    60.013, 60.488, 99.946, 97.641, 96.574, 91.606, 100, 99.327, 99.958, 93.907, 98.113,
    77.614, 99.432, 86.369, 96.951, 100, 90.661, 99.939, 95.935, 82.203, 99.334, 57.959,
    99.878, 91.955, 99.886, 100, 99.372, 100, 99.443, 99.373, 91.685, 95.413
]

# Creazione DataFrame
df = pd.DataFrame({
    "Sequence": sequences,
    "HOTA": hota,
    "MOTA": mota,
    "IDF1": idf1
})

# Plot
plt.figure(figsize=(14, 6))
plt.plot(df["Sequence"], df["HOTA"], label="HOTA", marker='o')
plt.plot(df["Sequence"], df["MOTA"], label="MOTA", marker='s')
plt.plot(df["Sequence"], df["IDF1"], label="IDF1", marker='^')
plt.axhline(80, color='gray', linestyle='--', linewidth=0.5)
plt.xticks(rotation=90)
plt.title("Metriche HOTA, MOTA, IDF1 per sequenza - KalmanTracker")
plt.ylabel("Score (%)")
plt.xlabel("Sequenza")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
