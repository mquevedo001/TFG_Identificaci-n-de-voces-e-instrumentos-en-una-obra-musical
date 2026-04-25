from pathlib import Path
import shutil
import random

random.seed(42)

root = Path("datasets/mix_generation")
source_folder = root / "foreground"

train_folder = root / "foreground_train"
valid_folder = root / "foreground_valid"

sources = ["vocals", "drums", "bass", "other"]

# Obtener lista de canciones (usamos vocals como referencia)
song_files = list((source_folder / "vocals").glob("*.wav"))
song_names = [f.stem for f in song_files]

random.shuffle(song_names)

split_idx = int(0.8 * len(song_names))
train_songs = song_names[:split_idx]
valid_songs = song_names[split_idx:]

print(f"Total songs: {len(song_names)}")
print(f"Train: {len(train_songs)}")
print(f"Valid: {len(valid_songs)}")

for split_name, split_songs, dest_root in [
    ("train", train_songs, train_folder),
    ("valid", valid_songs, valid_folder),
]:
    for source in sources:
        (dest_root / source).mkdir(parents=True, exist_ok=True)

    for song in split_songs:
        for source in sources:
            src_file = source_folder / source / f"{song}.wav"
            dst_file = dest_root / source / f"{song}.wav"
            if src_file.exists():
                shutil.move(str(src_file), str(dst_file))

print("Split completed.")