import matplotlib.pyplot as plt
import librosa
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
import librosa, librosa.display
import os

MAIN_LOSSES = ["l1","l2","l1_freq","l2_freq","logl1","logl2","log_mag","log_compressed_l2","lpsa","mask_l1"]
EXPERIMENTAL_LOSSES = ["deep_feature","deep_feature_emd"]
ADVANCED_LOSSES = ["l_mrs","lpsa_phase",]

def _is_audiosignal(x):
    return hasattr(x, "audio_data") and hasattr(x, "sample_rate")

def _to_numpy_and_sr(x, fallback_sr=None):
    """
    Devuelve (y, sr) con y en mono, tipo float32.
    x puede ser nussl.AudioSignal, np.ndarray o torch.Tensor.
    """
    # 1) sacar datos y sr
    if _is_audiosignal(x):
        y = x.audio_data  # nussl: (n_samples, n_channels)
        sr = int(x.sample_rate)
    else:
        # numpy/torch
        y = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
        sr = int(fallback_sr) if fallback_sr is not None else None

    # 2) asegurar 1D mono
    y = y.squeeze()
    if y.ndim == 2:
        # Caso común nussl: (n_samples, n_channels) -> media por canal
        if y.shape[0] >= y.shape[1]:
            y = y.mean(axis=1)
        else:
            # Si llegara como (channels, samples)
            y = y.mean(axis=0)
    y = y.astype(np.float32)

    return y, sr

def generateSourcesGraph(
    sources_dict,
    output_path,
    sr=None,                 # puedes pasar None si usas AudioSignal (se toma de ahí)
    n_fft=2048,
    hop_length=512,
    normalize_wave=False,
    db_range=(-80, 0),
):
    # --- Extraer arrays mono + sr por fuente
    stems = {}
    sr_detected = None

    for name, x in sources_dict.items():
        y, sr_x = _to_numpy_and_sr(x, fallback_sr=sr)
        if sr_x is None:
            raise ValueError("No se pudo determinar sample_rate; pásalo en sr o usa AudioSignal.")
        if sr_detected is None:
            sr_detected = sr_x
        elif sr_detected != sr_x:
            raise ValueError(f"Sample rates distintos entre fuentes: {sr_detected} vs {sr_x}")
        stems[name] = y

    sr = sr_detected

    # --- Igualar longitudes
    min_len = min(len(v) for v in stems.values())
    for k in stems:
        stems[k] = stems[k][:min_len]

    # --- Normalización opcional por RMS (para que la forma se diferencie por dinámica)
    if normalize_wave:
        for k, y in stems.items():
            rms = np.sqrt(np.mean(y**2) + 1e-12)
            stems[k] = y / (rms + 1e-12)

    names = list(stems.keys())
    n_sources = len(stems)

    # --- Figura: overlay + waveforms individuales + espectrogramas individuales
    fig = plt.figure(figsize=(14, 3 + 1.8*n_sources + 2.2*n_sources))
    gs = fig.add_gridspec(2*n_sources + 1, 1, hspace=0.5)

    # 1) Overlay waveform
    ax_overlay = fig.add_subplot(gs[0, 0])
    for name, y in stems.items():
        t = np.arange(len(y)) / sr
        ax_overlay.plot(t, y, alpha=0.6, linewidth=0.8, label=name)
    ax_overlay.set_title("Overlay waveform (todas las fuentes)")
    ax_overlay.set_ylabel("Amplitud")
    ax_overlay.set_xlabel("Tiempo (s)")
    ax_overlay.legend(loc='upper left', ncol=min(4, n_sources), fontsize=8, frameon=True)

    # 2) Waveforms por fuente
    for i, (name, y) in enumerate(stems.items(), start=1):
        ax = fig.add_subplot(gs[i, 0])
        t = np.arange(len(y)) / sr
        ax.plot(t, y, linewidth=0.8)
        ax.set_title(f"Waveform — {name}")
        ax.set_ylabel("Amp")
        if i == n_sources:
            ax.set_xlabel("Tiempo (s)")
        else:
            ax.set_xticklabels([])

    # 3) Espectrogramas por fuente (misma escala dB)
    S_list = []
    for y in stems.values():
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hann'))
        S_list.append(S)

    global_ref = max(S.max() for S in S_list) + 1e-12
    vmin, vmax = db_range

    for j, (name, S) in enumerate(zip(stems.keys(), S_list), start=n_sources+1):
        ax = fig.add_subplot(gs[j, 0])
        S_db = librosa.amplitude_to_db(S, ref=global_ref)
        img = librosa.display.specshow(
            S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz',
            vmin=vmin, vmax=vmax, ax=ax
        )
        ax.set_title(f"Espectrograma (dB) — {name}")
        cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB")
        cbar.ax.set_ylabel("dB", rotation=270, labelpad=12)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)



from pathlib import Path
from config import config

def save_sources(estimates, audio_signal, output_dir,audio_name=None):

    sources_dict = {}
    stems = {}

    num_sources = config.config['MODEL_NUM_SOURCES']
    num_sources = int(num_sources)

    if audio_name:
        audio_name = audio_name.replace('.wav','')
        output_dir = output_dir / f'{audio_name} stems'
    else:
        output_dir = output_dir / 'musdb_test_audios stems'

    output_dir.mkdir(parents=True ,exist_ok=True)

    print(f"[DEBUG] - Guardando los resultados en: {output_dir}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if num_sources == 2:
        # Vocal y acompañamiento
        stem_names = ['vocals', 'accompaniment']
        stems_dir = output_dir / 'stems'
        stems_dir.mkdir(parents=True, exist_ok=True)

        for est, name in zip(estimates, stem_names):
            stem_path = stems_dir / f"{name}.wav"
            est.write_audio_to_file(stem_path)
            sources_dict[name] = est
            stems[name] = str(stem_path)

    elif num_sources == 4:
        # Orden por convención MUSDB: vocals, drums, bass, other
        stem_names = ['vocals', 'bass', 'drums', 'other']
        stems_dir = output_dir / 'stems'
        stems_dir.mkdir(parents=True, exist_ok=True)

        for est, name in zip(estimates, stem_names):
            stem_path = stems_dir / f"{name}.wav"
            est.write_audio_to_file(stem_path)
            sources_dict[name] = est
            stems[name] = str(stem_path)

    else:
        num_sources_estimates = len(estimates)
        raise ValueError(f"Se esperaban 2 o 4 fuentes, pero el modelo tiene {num_sources_estimates}.")

    return stems, sources_dict

def get_model_full_path_from_loss_func(folder_path,loss_func:str):

    if (loss_func not in MAIN_LOSSES) and (loss_func not in EXPERIMENTAL_LOSSES) and (loss_func not in ADVANCED_LOSSES): raise ValueError("La función de pérdida no está en ningún grupo.")
    else:

        if loss_func in MAIN_LOSSES: return os.path.join(folder_path,'main')           
        elif loss_func in EXPERIMENTAL_LOSSES: return os.path.join(folder_path,'experimental')          
        elif loss_func in ADVANCED_LOSSES: return os.path.join(folder_path,'advanced')
            
    