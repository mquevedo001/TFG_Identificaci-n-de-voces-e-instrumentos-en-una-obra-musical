from nussl.datasets import transforms as nussl_tfm
from pathlib import Path
from config import config
from common import data,utils
import torch
import torchaudio


class AddMixtureMagnitude:
    def __init__(self, n_fft=1024, hop_length=256, win_length=1024):
        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=n_fft, hop_length=hop_length, win_length=win_length, power=2.0
        )

    def __call__(self, item):
        mix_audio = item['mix'].audio_data
        mix_tensor = torch.tensor(mix_audio).mean(dim=0, keepdim=True)
        mag_spec = self.stft(mix_tensor)
        item['mixture_magnitude'] = mag_spec.squeeze(0)
        return item


class AddMixturePhase:
    def __init__(self, n_fft=1024, hop_length=256, win_length=1024):
        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=n_fft, hop_length=hop_length, win_length=win_length, power=None
        )

    def __call__(self, item):
        mix_audio = item['mix'].audio_data
        mix_tensor = torch.tensor(mix_audio).mean(dim=0, keepdim=True)
        complex_spec = self.stft(mix_tensor)
        phase = torch.angle(complex_spec)
        item['mixture_phase'] = phase.squeeze(0)
        return item


def get_data(stft_params, max_mixtures = config.config['MAX_MIXTURES'], coherent_prob = config.config['COHERENT_PROB']):

    utils.logger()

    train_folder = '/home/martin/PycharmProjects/TFG/datasets/mix_generation/foreground'
    val_folder = "~/.nussl/tutorial/valid"

    print(f"Path:{Path('.')}")
    print(f"Training model with training mixes in in: {train_folder} ")
    print(f"Validating data with validation mixes in: {val_folder} ")

    if config.config['MODEL_NUM_SOURCES'] == 2:
        tfm = nussl_tfm.Compose([
            AddMixtureMagnitude(
                n_fft=stft_params.window_length,
                hop_length=stft_params.hop_length,
                win_length=stft_params.window_length
            ),
            AddMixturePhase(
                n_fft=stft_params.window_length,
                hop_length=stft_params.hop_length,
                win_length=stft_params.window_length
            ),
            nussl_tfm.SumSources([['vocals'], ['bass', 'drums', 'other']]),
            nussl_tfm.MagnitudeSpectrumApproximation(),
            nussl_tfm.ToSeparationModel(),
        ])

        train_data = data.on_the_fly(
            stft_params,
            transform=tfm,
            fg_path=train_folder,
            num_mixtures=max_mixtures,
            coherent_prob=coherent_prob
        )

        val_data = data.on_the_fly(
            stft_params,
            transform=tfm,
            fg_path=val_folder,
            num_mixtures=10,
            coherent_prob=coherent_prob
        )

    else:
        tfm = nussl_tfm.Compose([
            AddMixtureMagnitude(
                n_fft=stft_params.window_length,
                hop_length=stft_params.hop_length,
                win_length=stft_params.window_length
            ),
            AddMixturePhase(
                n_fft=stft_params.window_length,
                hop_length=stft_params.hop_length,
                win_length=stft_params.window_length
            ),
            nussl_tfm.MagnitudeSpectrumApproximation(),
            nussl_tfm.ToSeparationModel(),
        ])

        train_data = data.on_the_fly(
            stft_params,
            transform=tfm,
            fg_path=train_folder,
            num_mixtures=max_mixtures,
            coherent_prob=coherent_prob,
            sources=['bass', 'drums', 'vocals', 'other']
        )

        val_data = data.on_the_fly(
            stft_params,
            transform=tfm,
            fg_path=val_folder,
            num_mixtures=10,
            coherent_prob=coherent_prob,
            sources=['bass', 'drums', 'vocals', 'other']
        )

    return train_data, val_data
