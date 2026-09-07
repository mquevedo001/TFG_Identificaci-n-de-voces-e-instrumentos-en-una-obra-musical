from pathlib import Path


def load_test_dataset(root):
    root = Path(root)

    tracks = []

    for track_dir in sorted(root.iterdir()):
        if not track_dir.is_dir():continue
            
        required = ["mixture.wav","vocals.wav","bass.wav","drums.wav","other.wav"]
        missing = [filename for filename in required if not (track_dir / filename).exists()]

        if missing:raise FileNotFoundError(f"Faltan archivos en {track_dir}: {missing}")

        tracks.append({
            "mixture": track_dir / "mixture.wav",
            "sources": {
                "vocals": track_dir / "vocals.wav",
                "bass": track_dir / "bass.wav",
                "drums": track_dir / "drums.wav",
                "other": track_dir / "other.wav",
            },
        })

    return tracks

