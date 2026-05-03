from pathlib import Path


def load_test_dataset(root):
    tracks = []

    for track_dir in Path(root).iterdir():
        item = {
            "mixture": track_dir / "mixture.wav",
            "sources": {
                "vocals": track_dir / "vocals.wav",
                "bass": track_dir / "bass.wav",
                "drums": track_dir / "drums.wav",
                "other": track_dir / "other.wav",
            }
        }
        tracks.append(item)

    return tracks


