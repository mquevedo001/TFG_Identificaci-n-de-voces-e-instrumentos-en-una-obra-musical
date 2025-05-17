import random

from common import viz,data
import nussl
from pathlib import Path
from utils.config import config


stft_params = nussl.STFTParams(
    window_length = config.config['STFT_WINDOW_LENGTH'],
    hop_length = config.config['STFT_HOP_LENGTH'],
    window_type = config.config['STFT_WINDOW_TYPE'],
)

def deploy(output_dir, audio_path = None):
    # Cargar modelo
    output_folder = Path('.') / 'models' / 'checkpoints'
    model_path = output_folder / 'latest.model.pth'

    separator = nussl.separation.deep.DeepMaskEstimation(
        nussl.AudioSignal(),
        model_path=str(model_path.resolve()),
        device=config.config['DEVICE'],
    )

    # Cargar audio
    if audio_path != None:
        audio_signal = nussl.AudioSignal(audio_path)
        separator.audio_signal = audio_signal
        estimates = separator()

        residual = audio_signal - estimates[0]
        residual.stft()
        estimates.append(residual)
    else:
        test_folder = "~/.nussl/tutorial/test/"
        test_data = data.mixer(stft_params, transform=None,
                               fg_path=test_folder, num_mixtures=config.config['MAX_MIXTURES'], coherent_prob=1.0)
        item = test_data[0]

        separator.audio_signal = item['mix']
        estimates = separator()
        # Since our model only returns one source, let's tack on the
        # residual (which should be accompaniment)
        estimates.append(item['mix'] - estimates[0])

        viz.show_sources(estimates)

        audio_signal = estimates[0]



    # Alinear tamaños
    estimates = align_estimates(estimates)

    # Crear sources dict
    sources_dict = {
        f'Source {i}': estimate for i, estimate in enumerate(estimates)
    }

    # Crear carpetas
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'source0').mkdir(exist_ok=True)
    (output_dir / 'source1').mkdir(exist_ok=True)

    # Guardar fuentes
    estimates[0].write_audio_to_file(output_dir / 'source0' / 'source0.wav')
    estimates[1].write_audio_to_file(output_dir / 'source1' / 'source1.wav')

    # Guardar el gráfico
    fig = viz.show_sources(sources_dict, return_figure=True)
    fig.savefig(output_dir / 'sources_plot.png')
    import matplotlib.pyplot as plt
    plt.close(fig)

    # Copiar audio original
    audio_signal.write_audio_to_file(output_dir / 'original.wav')

    print(f"Resultados guardados en {output_dir}")



def align_estimates(estimates):
    min_length = min([s.signal_length for s in estimates])
    for i in range(len(estimates)):
        estimates[i].truncate_samples(min_length)
    return estimates