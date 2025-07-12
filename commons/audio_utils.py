import nussl
from pathlib import Path
from common import data
from config import config
import torch




def spectral_convergence(y_hat, y, eps=1e-8):
    num = torch.norm(torch.abs(y_hat) - torch.abs(y), p='fro')
    denom = torch.norm(torch.abs(y), p='fro') + eps
    return num / denom


def align_estimates(estimates):
    min_length = min([s.signal_length for s in estimates])
    for est in estimates:
        est.truncate_samples(min_length)
    return estimates


def load_model(model_name='best.model.pth'):

    loss_fn = config.config['MODEL_LOSS_FUNCTION']
    num_sources = config.config['MODEL_NUM_SOURCES']

    model_path = Path('.') / 'checkpoints'/f'{loss_fn} checkpoints' / f'{num_sources}stems' /'best.model.pth'


    separator = nussl.separation.deep.DeepMaskEstimation(
        nussl.AudioSignal(),
        model_path= model_path,
        device=config.config['DEVICE']
    )
    return separator

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
