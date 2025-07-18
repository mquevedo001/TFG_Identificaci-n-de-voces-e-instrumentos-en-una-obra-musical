import nussl
from pathlib import Path
from common import data
from config import config
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
    loss_fn = config.config['MODEL_LOSS_FUNCTION']
    num_sources = config.config['MODEL_NUM_SOURCES']
    model_path = Path('.') / 'checkpoints' / 'Mis_modelos' / f'{loss_fn} checkpoints' / f'{num_sources}stems' / 'best.model.pth'

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

    model.load_state_dict(torch.load(model_path, map_location=config.config['DEVICE']))
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
