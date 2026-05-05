from common import data
from config import config
from pathlib import Path
import sys
import numpy as np
import torch
import os
import re
from models.Mi_modelo.mask_inference import MaskInference


def spectral_convergence(y_hat, y, eps=1e-8):
    num = torch.norm(torch.abs(y_hat) - torch.abs(y), p='fro')
    denom = torch.norm(torch.abs(y), p='fro') + eps
    return num / denom


def align_estimates(estimates):
    min_length = min([s.signal_length for s in estimates])
    for est in estimates:
        est.truncate_samples(min_length)
    return estimates


def build_complex_from_mag_and_phase(magnitude, mixture_phase):
    """
    magnitude:     [B, T, F, C, S]
    mixture_phase: [B, F, T]

    devuelve:      [B, S, C, F, T] complejo
    """
    if magnitude.dim() != 5:
        raise ValueError(f"magnitude shape inesperada: {magnitude.shape}")
    if mixture_phase.dim() != 3:
        raise ValueError(f"mixture_phase shape inesperada: {mixture_phase.shape}")

    # [B, F, T] -> [B, T, F]
    phase = mixture_phase.permute(0, 2, 1)

    # Alinear T y F por seguridad
    T = min(magnitude.shape[1], phase.shape[1])
    F = min(magnitude.shape[2], phase.shape[2])

    magnitude = magnitude[:, :T, :F, :, :]      # [B, T, F, C, S]
    phase = phase[:, :T, :F]                    # [B, T, F]

    # Expandir fase a [B, T, F, 1, 1]
    phase = phase.unsqueeze(-1).unsqueeze(-1)

    # Construir espectro complejo
    complex_spec = torch.polar(magnitude, phase)   # [B, T, F, C, S]

    # Reordenar a [B, S, C, F, T] para facilitar ISTFT
    complex_spec = complex_spec.permute(0, 4, 3, 2, 1).contiguous()

    return complex_spec

def reconstruct_waveforms_from_mag_phase(magnitude, mixture_phase):
    """
    magnitude:     [B, T, F, C, S]
    mixture_phase: [B, F, T]

    devuelve:      [B, S, C, N]
    """
    complex_spec = build_complex_from_mag_and_phase(magnitude, mixture_phase)

    B, S, C, F, T = complex_spec.shape

    n_fft = config.config['STFT_WINDOW_LENGTH']
    hop_length = config.config['STFT_HOP_LENGTH']
    win_length = config.config['STFT_WINDOW_LENGTH']

    window = torch.hann_window(win_length, device=complex_spec.device)

    waveforms = []

    for b in range(B):
        batch_sources = []
        for s in range(S):
            source_channels = []
            for c in range(C):
                spec = complex_spec[b, s, c]  # [F, T]
                wav = torch.istft(
                    spec,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    window=window,
                    center=True,
                    normalized=False,
                    onesided=True,
                    return_complex=False
                )
                source_channels.append(wav)
            source_channels = torch.stack(source_channels, dim=0)  # [C, N]
            batch_sources.append(source_channels)
        batch_sources = torch.stack(batch_sources, dim=0)  # [S, C, N]
        waveforms.append(batch_sources)

    waveforms = torch.stack(waveforms, dim=0)  # [B, S, C, N]
    return waveforms

# -----------------------------
# CHECKPOINT SELECTION
# -----------------------------

def extract_val_loss(file):
    match = re.search(r'val_loss=([-+]?\d*\.\d+|\d+)', file)
    if match:
        return float(match.group(1))
    return float('inf')


def load_best_model(stems_folder_model_path):
    checkpoint_names = [f for f in os.listdir(stems_folder_model_path)]
    return min(checkpoint_names, key=extract_val_loss)


# -----------------------------
# DEBUG HELPERS
# -----------------------------

def print_checkpoint_keys(checkpoint):
    print("\n[CHECKPOINT KEYS]")
    if isinstance(checkpoint, dict):
        for k in checkpoint.keys():
            print(k)
    else:
        print("Checkpoint no es dict:", type(checkpoint))


def get_state_dict_from_checkpoint(checkpoint):
    """
    Soporta:
    - {"model": state_dict}
    - {"state_dict": state_dict}
    - state_dict plano
    """
    if isinstance(checkpoint, dict):

        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            return checkpoint["model"]

        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]

        return checkpoint

    raise ValueError("Checkpoint no válido")


def infer_arch_from_state_dict(state_dict):
    """
    Intenta inferir el hidden_size real del checkpoint inspeccionando las formas de los pesos de la RNN.
    intenta inferir hidden_size real del checkpoint
    """
    for k, v in state_dict.items():
        if "rnn.weight_ih_l0" in k:
            hidden = v.shape[0]
            bidir_factor = 2 if "reverse" in str(state_dict.keys()) else 1
            print("\n[CHECKPOINT ARCH INFERENCE]")
            print("RNN hidden_size approx:", hidden // bidir_factor)
            return hidden // bidir_factor
    return None


def print_model_parameter_norms(model, max_print=5):
    print("\n[MODEL INIT DEBUG]")

    total_norm = 0
    count = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            norm = param.data.norm().item()
            total_norm += norm
            count += 1

            if count <= max_print:
                print(f"{name}: norm={norm:.6f}")

    print("AVG PARAM NORM:", total_norm / max(count, 1))


def print_load_report(missing, unexpected):
    print("\n[LOAD REPORT]")

    if missing:
        print(f"[WARN] Missing keys: {len(missing)}")
        for k in missing[:10]:
            print("  -", k)

    if unexpected:
        print(f"[WARN] Unexpected keys: {len(unexpected)}")
        for k in unexpected[:10]:
            print("  -", k)


def compare_shapes(model, state_dict):
    print("\n[SHAPE CHECK]")

    mismatches = 0

    for name, param in model.named_parameters():
        if name in state_dict:
            if param.shape != state_dict[name].shape:
                print(f"[MISMATCH] {name}")
                print(f"   model:     {param.shape}")
                print(f"   checkpoint:{state_dict[name].shape}")
                mismatches += 1

    if mismatches == 0:
        print("No shape mismatches detected before loading")


# -----------------------------
# MAIN LOAD FUNCTION
# -----------------------------

def load_model(loss_fn, num_sources):

    sys.modules.setdefault("numpy._core", np)

    model_path = (
        Path('.') / 'checkpoints' / 'Mis_modelos' /
        f'{loss_fn} checkpoints' / f'{num_sources}stems'
    )

    best_model_tag = load_best_model(model_path)
    model_path = model_path / best_model_tag

    if not os.path.exists(model_path):
        raise FileNotFoundError(f'No existe la ruta: {model_path}')

    print("\n[MODEL PATH]")
    print(repr(str(model_path)))

    # -----------------------------
    # LOAD CHECKPOINT FIRST 
    # -----------------------------
    checkpoint = torch.load(
        model_path,
        map_location=config.config['DEVICE'],
        weights_only=False
    )

    print_checkpoint_keys(checkpoint)

    state_dict = get_state_dict_from_checkpoint(checkpoint)

    # -----------------------------
    # ARCH INFERENCE
    # -----------------------------
    inferred_hidden = infer_arch_from_state_dict(state_dict)

    if inferred_hidden is not None:
        hidden_size = inferred_hidden
    else:
        hidden_size = config.config['MODEL_HIDDEN_SIZE']

    nf = config.config['STFT_WINDOW_LENGTH'] // 2 + 1

    # -----------------------------
    # MODEL BUILD
    # -----------------------------
    model = MaskInference.build(
        nf,
        num_audio_channels=config.config['MODEL_NUM_CHANNELS'],
        hidden_size=hidden_size,
        num_layers=config.config['MODEL_NUM_LAYERS'],
        bidirectional=config.config['MODEL_BIDIRECTIONAL'],
        dropout=config.config['MODEL_DROPOUT'],
        num_sources=num_sources,
        activation=config.config['MODEL_ACTIVATION'],
    )

    print("\n[MODEL CONFIG USED]")
    print("hidden_size:", hidden_size)
    print("bidirectional:", config.config['MODEL_BIDIRECTIONAL'])
    print("num_layers:", config.config['MODEL_NUM_LAYERS'])

    # -----------------------------
    # DEBUG BEFORE LOAD
    # -----------------------------
    compare_shapes(model, state_dict)
    print_model_parameter_norms(model)

    # -----------------------------
    # LOAD WEIGHTS
    # -----------------------------
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print_load_report(missing, unexpected)

    model.eval()
    return model


def conseguirAudioDatabase(test_folder,stft_params):

    test_data = data.mixer(
       stft_params,
       transform=None,
       fg_path=test_folder,
       num_mixtures=config.config['MAX_MIXTURES'],
       coherent_prob=1.0
    )

    item = test_data[1]
    audio_signal = item['mix']

    return audio_signal
