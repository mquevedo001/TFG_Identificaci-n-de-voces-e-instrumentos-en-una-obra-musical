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
