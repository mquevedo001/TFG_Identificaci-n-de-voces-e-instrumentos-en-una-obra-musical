from common import viz
import matplotlib.pyplot as plt
from config import config
from pathlib import Path


def generateSourcesGraph(sources_dict,output_dir):
    fig = viz.show_sources(sources_dict, return_figure=True)
    fig.savefig(output_dir / 'sources_plot.png')
    plt.close(fig)


def save_sources(estimates,audio_signal,output_dir):

    sources_dict = {}
    stems = {}
    num_sources = config.config['MODEL_NUM_SOURCES']
    if num_sources == 2:
        # Vocal y acompañamiento
        vocal = estimates[0]
        accompaniment = audio_signal - vocal

        vocal_path = Path('.') / 'Results' / 'stems' / 'vocals.wav'
        accompaniment_path = Path('.') / 'Results' / 'stems' / 'accompaniment.wav'
        #/home/martin/PycharmProjects/TFG/Results/stems/accompaniment.wav
        vocal.write_audio_to_file(vocal_path)
        accompaniment.write_audio_to_file(accompaniment_path)

        sources_dict['vocals'] = vocal
        sources_dict['accompaniment'] = accompaniment

        stems['vocals'] = str(vocal_path)
        stems['accompaniment'] = str(accompaniment_path)

    elif num_sources == 4:
        # Orden por convención MUSDB: vocals, drums, bass, other
        stem_names = ['vocals', 'drums', 'bass', 'other']
        for est, name in zip(estimates, stem_names):
            stem_path = output_dir /'stems'/ f"{name}.wav"
            est.write_audio_to_file(stem_path)
            sources_dict[name] = est
            stems[name] = str(stem_path)

    else:
        raise ValueError(f"Se esperaban 2 o 4 fuentes, pero el modelo tiene {num_sources}.")

    return stems,sources_dict