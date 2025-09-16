import os

seq_names = [
    "ArmyDiver1", "Ballena", "BlueFish1", "BlueFish2", "BoySwimming", "CenoteAngelita",
    "DeepSeaFish", "Dolphin1", "Dolphin2", "Fisherman", "FishFollowing", "GarryFish",
    "HoverFish1", "HoverFish2", "JerkbaitBites", "MonsterCreature1", "MonsterCreature2",
    "Octopus1", "Octopus2", "PinkFish", "SeaDiver", "SeaDragon", "SeaTurtle1",
    "SeaTurtle2", "SeaTurtle3", "Steinlager", "WhaleAtBeach1", "WhaleAtBeach2",
    "WhaleDiving", "WhiteShark"
]

output_dir = "C:/Users/fedem/Desktop/TrackEval-master/TrackEval-master/data/CustomDataset/seqmaps"
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, "CustomDataset-train.txt"), "w") as f:
    f.write("name\n")
    for name in seq_names:
        f.write(name + "\n")
