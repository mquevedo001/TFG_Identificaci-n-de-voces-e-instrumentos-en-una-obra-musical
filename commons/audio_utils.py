import nussl
import os
from pathlib import Path
from common import data
from config import config







def align_estimates(estimates):
    min_length = min([s.signal_length for s in estimates])
    for est in estimates:
        est.truncate_samples(min_length)
    return estimates


def load_model(model_name='best.model.pth'):


    stem_count = config.config['MODEL_NUM_SOURCES']
    checkpoint_path = Path(f'checkpoints/{stem_count}stems') / model_name

    separator = nussl.separation.deep.DeepMaskEstimation(
        nussl.AudioSignal(),
        model_path=str(checkpoint_path.resolve()),
        device=config.config['DEVICE']
    )
    return separator

def conseguirAudioDatabase(test_folder,stft_params):
    if not os.listdir(test_folder):
        test_data = data.mixer(stft_params, transform=None,
                               fg_path=test_folder,
                               num_mixtures=config.config['MAX_MIXTURES'],
                               coherent_prob=1.0)
    item = test_data[0]
    audio_signal = item['mix']

    return audio_signal
