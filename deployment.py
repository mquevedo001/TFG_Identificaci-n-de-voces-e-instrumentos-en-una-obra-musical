import random
import nussl
from pathlib import Path
from utils.config import config
from common import viz, data
import matplotlib.pyplot as plt
import os

stft_params = nussl.STFTParams(
    window_length=config.config['STFT_WINDOW_LENGTH'],
    hop_length=config.config['STFT_HOP_LENGTH'],
    window_type=config.config['STFT_WINDOW_TYPE'],
)

def deploy(output_dir=None, audio_path=None):
    # Cargar modelo
    output_folder = Path('.') / 'checkpoints'
    model_path = output_folder / 'best.model.pth'

    separator = nussl.separation.deep.DeepMaskEstimation(
        nussl.AudioSignal(),
        model_path=str(model_path.resolve()),
        device=config.config['DEVICE'],
    )

    # Cargar audio
    if audio_path is not None:
        audio_signal = nussl.AudioSignal(audio_path)
    else:
        test_folder = "~/.nussl/tutorial/test/"
        test_data = data.mixer(stft_params, transform=None,
                               fg_path=test_folder,
                               num_mixtures=config.config['MAX_MIXTURES'],
                               coherent_prob=1.0)
        item = test_data[0]
        audio_signal = item['mix']

    separator.audio_signal = audio_signal
    estimates = separator()

    print("Número de fuentes separadas:", len(estimates))
    for i, est in enumerate(estimates):
        print(f"Fuente {i} RMS: {est.rms()}, Duración: {est.signal_length}")

    # Alinear estimaciones
    estimates = align_estimates(estimates)

    # Crear carpeta de salida
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sources_dict = {}
    stems = {}
    if(output_dir != None):
        for i, estimate in enumerate(estimates):


            est_dir = output_dir / f'source{i}'
            est_dir.mkdir(exist_ok=True)
            file_path = est_dir / f'source{i}.wav'
            estimate.write_audio_to_file(file_path)
            sources_dict[f'Source {i}'] = estimate

            name = estimate.path_to_file or f'stem_{i}.wav'
            stem_path = output_folder / f'stem_{i}.wav'
            estimate.write_audio_to_file(stem_path)
            stems[f'stem_{i}'] = stem_path

        # Graficar estimaciones
        fig = viz.show_sources(sources_dict, return_figure=True)
        fig.savefig(output_dir / 'sources_plot.png')
        plt.close(fig)

        # Guardar mezcla original
        audio_signal.write_audio_to_file(output_dir / 'original.wav')
        print(f"Resultados guardados en {output_dir}")


    return stems




def align_estimates(estimates):
    min_length = min([s.signal_length for s in estimates])
    for est in estimates:
        est.truncate_samples(min_length)
    return estimates
