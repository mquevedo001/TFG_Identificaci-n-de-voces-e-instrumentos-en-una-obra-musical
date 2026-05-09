from pathlib import Path
import numpy as np
import nussl

# =========================
# RUTAS
# =========================

TRACK = Path(
    r"C:\Users\rdpuser\TFG_Identificaci-n-de-voces-e-instrumentos-en-una-obra-musical\datasets\representative_test\track_00"
)

OUT = Path(
    r"C:\Users\rdpuser\TFG_Identificaci-n-de-voces-e-instrumentos-en-una-obra-musical\Results\Separation tests\logl1\4stems\mixture stems\stems"
)

names = ["vocals", "bass", "drums", "other"]


# =========================
# CARGA MONO
# =========================

def mono(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"No existe: {path}")

    sig = nussl.AudioSignal(str(path))

    # audio_data shape:
    # (channels, samples)
    if sig.audio_data.ndim == 1:
        y = sig.audio_data
    else:
        y = sig.audio_data.mean(axis=0)

    return y.astype(np.float64)


# =========================
# REFERENCIAS
# =========================

refs = {}

for name in names:
    p = TRACK / f"{name}.wav"
    print(f"[REF] {p}")
    refs[name] = mono(p)


# =========================
# ESTIMACIONES
# =========================

ests = {}

for name in names:
    p = OUT / f"{name}.wav"
    print(f"[EST] {p}")
    ests[name] = mono(p)


# =========================
# MATRIZ DE CORRELACIÓN
# =========================

print("\n[CORRELATION MATRIX]")
print("rows = estimates, cols = references")
print("              " + " ".join(f"{n:>10}" for n in names))

for est_name in names:

    row = []

    for ref_name in names:

        n = min(len(ests[est_name]), len(refs[ref_name]))

        a = ests[est_name][:n]
        b = refs[ref_name][:n]

        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            corr = np.nan
        else:
            corr = np.corrcoef(a, b)[0, 1]

        row.append(corr)

    print(f"{est_name:>12} " + " ".join(f"{v:10.4f}" for v in row))