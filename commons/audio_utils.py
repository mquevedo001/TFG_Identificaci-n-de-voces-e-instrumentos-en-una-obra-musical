from common import data
from config import config
from pathlib import Path
import sys
import numpy as np
import torch
import os
import re
from models.Mi_modelo.mask_inference import MaskInference
from commons.experiment_utils import resolve_checkpoint_dir

def spectral_convergence(y_hat, y, eps=1e-8):
    num = torch.norm(torch.abs(y_hat) - torch.abs(y), p='fro')
    denom = torch.norm(torch.abs(y), p='fro') + eps
    return num / denom


def align_estimates(estimates):
    min_length = min([s.signal_length for s in estimates])
    for est in estimates:
        est.truncate_samples(min_length)
    return estimates

def print_stem_diagnostics(audio_signal, sources_dict):
    import numpy as np

    mix = audio_signal.audio_data.squeeze()
    if mix.ndim == 2:
        mix = mix.mean(axis=0) if mix.shape[0] < mix.shape[1] else mix.mean(axis=1)

    print("\n[STEM DIAGNOSTICS]")
    for name, sig in sources_dict.items():
        y = sig.audio_data.squeeze()
        if y.ndim == 2:
            y = y.mean(axis=0) if y.shape[0] < y.shape[1] else y.mean(axis=1)

        n = min(len(mix), len(y))
        mix_n = mix[:n]
        y_n = y[:n]

        rms = float(np.sqrt(np.mean(y_n ** 2) + 1e-12))
        corr = float(np.corrcoef(mix_n, y_n)[0, 1])

        print(f"{name}: rms={rms:.6f}, corr_con_mix={corr:.4f}")

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
    checkpoint_names = [
        f for f in os.listdir(stems_folder_model_path)
        if f.endswith(".pt")
    ]

    if not checkpoint_names:
        raise FileNotFoundError(
            f"No hay checkpoints .pt en {stems_folder_model_path}"
        )

    # En train.py se guarda score_function = -val_loss.
    # Por tanto, el mejor checkpoint es el de mayor score,
    return max(checkpoint_names, key=extract_val_loss)


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
    Infiere arquitectura real desde los pesos guardados.

    Para LSTM:
      weight_ih_l0: [4 * hidden_size, input_size]
      weight_hh_l0: [4 * hidden_size, hidden_size]

    La forma más fiable de sacar hidden_size es weight_hh_l0.shape[1].
    """

    weight_ih_key = None
    weight_hh_key = None

    for k in state_dict.keys():
        if "recurrent_stack.rnn.weight_ih_l0" in k and "reverse" not in k:
            weight_ih_key = k
        if "recurrent_stack.rnn.weight_hh_l0" in k and "reverse" not in k:
            weight_hh_key = k

    if weight_ih_key is None or weight_hh_key is None:
        raise ValueError(
            "No se han encontrado pesos RNN l0 en el checkpoint. "
            "No puedo inferir la arquitectura."
        )

    input_size = int(state_dict[weight_ih_key].shape[1])
    hidden_size = int(state_dict[weight_hh_key].shape[1])
    gate_rows = int(state_dict[weight_ih_key].shape[0])
    gate_multiplier = gate_rows // hidden_size

    bidirectional = any(
        "recurrent_stack.rnn.weight_ih_l0_reverse" in k
        for k in state_dict.keys()
    )

    layer_ids = []
    for k in state_dict.keys():
        match = re.search(r"recurrent_stack\.rnn\.weight_ih_l(\d+)", k)
        if match:
            layer_ids.append(int(match.group(1)))

    num_layers = max(layer_ids) + 1 if layer_ids else 1

    num_channels = int(config.config["MODEL_NUM_CHANNELS"])
    if input_size % num_channels != 0:
        raise ValueError(
            f"input_size={input_size} no divisible por num_channels={num_channels}"
        )

    num_features = input_size // num_channels

    print("\n[CHECKPOINT ARCH INFERENCE]")
    print("input_size:", input_size)
    print("num_features:", num_features)
    print("hidden_size:", hidden_size)
    print("gate_multiplier:", gate_multiplier)
    print("bidirectional:", bidirectional)
    print("num_layers:", num_layers)

    return {
        "num_features": num_features,
        "hidden_size": hidden_size,
        "bidirectional": bidirectional,
        "num_layers": num_layers,
    }


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

def load_model(loss_fn, num_sources,checkpoint_path):

    sys.modules.setdefault("numpy._core", np)

    model_path = resolve_checkpoint_dir(loss_fn, num_sources)

    best_model_tag = load_best_model(model_path)
    model_path = model_path / best_model_tag

    if not os.path.exists(model_path):
        raise FileNotFoundError(f'No existe la ruta: {model_path}')

    print("\n[MODEL PATH]")
    print(repr(str(model_path)))

    # -----------------------------
    # LOAD CHECKPOINT FIRST 
    # -----------------------------

    if checkpoint is None:
        checkpoint = torch.load(
            model_path,
            map_location=config.config['DEVICE'],
            weights_only=True
        )
    else:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=config.config['DEVICE'],
            weights_only=True
        )
    

    print_checkpoint_keys(checkpoint)

    state_dict = get_state_dict_from_checkpoint(checkpoint)

    # -----------------------------
    # ARCH INFERENCE
    # -----------------------------
    inferred_hidden = infer_arch_from_state_dict(state_dict)

    hidden_size = inferred_hidden['hidden_size']
    nf = inferred_hidden['num_features']
    bidirectional = inferred_hidden['bidirectional']
    # -----------------------------
    # MODEL BUILD
    # -----------------------------
    model = MaskInference.build(
        nf,
        num_audio_channels=config.config['MODEL_NUM_CHANNELS'],
        hidden_size=hidden_size,
        num_layers=config.config['MODEL_NUM_LAYERS'],
        bidirectional=bidirectional,
        dropout=config.config['MODEL_DROPOUT'],
        num_sources=num_sources,
        activation=config.config['MODEL_ACTIVATION'],
    )

    print("\n[MODEL CONFIG USED]")
    print("hidden_size:", hidden_size)
    print("bidirectional:", bidirectional)
    print("num_layers:", config.config['MODEL_NUM_LAYERS'])

    # -----------------------------
    # DEBUG BEFORE LOAD
    # -----------------------------
    compare_shapes(model, state_dict)
    print_model_parameter_norms(model)

    # -----------------------------
    # LOAD WEIGHTS
    # -----------------------------
    missing, unexpected = model.load_state_dict(state_dict, strict=True)

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
