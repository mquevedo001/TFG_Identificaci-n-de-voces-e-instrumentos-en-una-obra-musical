from common import viz
import nussl
from pathlib import Path
from utils.config import config


def deploy(audio_path, output_dir):
    # Cargar modelo
    output_folder = Path('.') / 'models' / 'checkpoints'
    model_path = output_folder / 'latest.model.pth'

    separator = nussl.separation.deep.DeepMaskEstimation(
        nussl.AudioSignal(),
        model_path=str(model_path.resolve()),
        device=config.config['DEVICE'],
    )

    # Cargar audio
    audio_signal = nussl.AudioSignal(audio_path)
    separator.audio_signal = audio_signal
    estimates = separator()

    residual = audio_signal - estimates[0]
    residual.stft()
    estimates.append(residual)

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