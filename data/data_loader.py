from nussl.datasets import transforms as nussl_tfm
from pathlib import Path
from config import config
from common import data,utils
import torch
import torchaudio
import os


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

class AddStackedSourcePhase:
    def __init__(self, source_order,n_fft=1024, hop_length=256, win_length=1024):
        self.source_order = source_order
        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            power=None
        )

    def __call__(self, item):
        phases = []

        for key in self.source_order:
            source_signal = item['sources'][key]
            src_audio = source_signal.audio_data
            src_tensor = torch.tensor(src_audio).mean(dim=0, keepdim=True)
            complex_spec = self.stft(src_tensor)
            phase = torch.angle(complex_spec).squeeze(0) # Esto es [F,T]
            phases.append(phase.squeeze(0))

        item['source_phase'] = torch.stack(phases) #Esto es [F,T,S]
        return item
    
class AddGroupedSourcePhase:
    def __init__(self,source_groups,n_fft=1024, hop_length=256, win_length=1024):
        self.source_groups = source_groups
        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            power=None
        )

    def __call__(self, item):
        group_phases = []

        for group in self.source_groups:
            complex_sum = None

            for key in group:
                source_signal = item['sources'][key]
                src_audio = source_signal.audio_data
                src_tensor = torch.tensor(src_audio).mean(dim=0, keepdim=True)
                complex_spec = self.stft(src_tensor)

                if complex_sum is None:
                    complex_sum = complex_spec
                else:
                    complex_sum += complex_spec

            group_phase = torch.angle(complex_sum).squeeze(0) # Esto es [F,T]
            group_phases.append(group_phase)

        item['source_phase'] = torch.stack(group_phases,dim=1) # Esto es [F,T,S]
        return item

def get_data(stft_params, max_mixtures=config.config['MAX_MIXTURES'], coherent_prob=config.config['COHERENT_PROB']):
    utils.logger()

    train_folder = r'C:\Users\rdpuser\TFG_Identificaci-n-de-voces-e-instrumentos-en-una-obra-musical\datasets\mix_generation\foreground_train'
    val_folder = r'C:\Users\rdpuser\TFG_Identificaci-n-de-voces-e-instrumentos-en-una-obra-musical\datasets\mix_generation\foreground_valid'
    num_sources = config.config['MODEL_NUM_SOURCES']

    print(f"Training model with training mixes in: {train_folder}")
    print(f"Validating data with validation mixes in: {val_folder}")
    print(f"[DEBUG] Num de sources: {num_sources}")

    
    if int(num_sources) == 2:
        print("[DEBUG]Pipeline de 2 fuentes")
        source_groups = [['vocals'], ['bass', 'drums', 'other']]
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
            AddGroupedSourcePhase(
                source_groups=source_groups,
                n_fft=stft_params.window_length,
                hop_length=stft_params.hop_length,
                win_length=stft_params.window_length
            ),
            nussl_tfm.SumSources([['vocals'], ['bass', 'drums', 'other']]),
            nussl_tfm.MagnitudeSpectrumApproximation(),
            nussl_tfm.ToSeparationModel(),
        ])

    else:
        print("[DEBUG] Pipeline de 4 fuentes")
        source_order = ['vocals', 'bass', 'drums', 'other']
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
            AddStackedSourcePhase(
                source_order=source_order,
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
    )

    val_data = data.on_the_fly(
        stft_params,
        transform=tfm,
        fg_path=val_folder,
        num_mixtures=200,
        coherent_prob=coherent_prob,
    )

    return train_data, val_data

