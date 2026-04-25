from common import data
from config import config
from pathlib import Path
import sys
import numpy as np
import torch
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

def load_model():
    # --- shim para checkpoints antiguos que referencian numpy._core ---
    sys.modules.setdefault("numpy._core", np)

    loss_fn = config.config['MODEL_LOSS_FUNCTION'].lower()
    num_sources = config.config['MODEL_NUM_SOURCES']

    # OJO con la carpeta con espacio: Path lo maneja bien.
    model_path = (
        Path('.') / 'checkpoints' / 'Mis_modelos' /
        f'{loss_fn} checkpoints' / f'{num_sources}stems' / 'checkpoints' / 'best.model.pth'
    )

    if not model_path.exists():
        raise FileNotFoundError(f'No encuentro el checkpoint en: {model_path}')

    nf = config.config['STFT_WINDOW_LENGTH'] // 2 + 1

    model = MaskInference.build(
        nf,
        num_audio_channels=config.config['MODEL_NUM_CHANNELS'],
        hidden_size=config.config['MODEL_HIDDEN_SIZE'],
        num_layers=config.config['MODEL_NUM_LAYERS'],
        bidirectional=config.config['MODEL_BIDIRECTIONAL'],
        dropout=config.config['MODEL_DROPOUT'],
        num_sources=num_sources,
        activation=config.config['MODEL_ACTIVATION']
    )

    # Cargar checkpoint nussl (puede venir con 'state_dict', 'config', 'metadata')
    checkpoint = torch.load(model_path, map_location=config.config['DEVICE'])

    # Extraer el state_dict correcto
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint  # por si fuese un state_dict plano

    # A veces los checkpoints tienen prefijos raros; si hiciera falta, aquí se podrían limpiar:
    # state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # Cargar pesos (usa strict=True primero; si falla por alguna clave cosmética, prueba strict=False)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print('[WARN] Claves faltantes al cargar:', missing)
    if unexpected:
        print('[WARN] Claves inesperadas al cargar:', unexpected)

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
