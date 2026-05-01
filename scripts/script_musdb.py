from pathlib import Path
import soundfile as sf
import musdb
import numpy as np
OUTPUT = Path("C:\\Users\\rdpuser\\TFG_Identificaci-n-de-voces-e-instrumentos-en-una-obra-musical\\datasets\\representative_test")
OUTPUT.mkdir(parents=True, exist_ok=True)

mus = musdb.DB(
    root="C:/datasets/musdb18",
    subsets="test",
    download="True"
)

selected_tracks = mus.tracks[:20]

for i, track in enumerate(selected_tracks):
    track_dir = OUTPUT / f"track_{i:02d}"
    track_dir.mkdir(exist_ok=True)

    mix = track.audio.astype(np.float32)

    sf.write(
        track_dir / "mixture.wav",
        mix,
        track.rate,
        format='WAV'
    )

    for name, source in track.sources.items():
        src = source.audio.astype(np.float32)

        sf.write(
            track_dir / f"{name}.wav",
            src,
            track.rate,
            format='WAV'
        )