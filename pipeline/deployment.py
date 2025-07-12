import nussl
from pathlib import Path
from config import config
from commons.audio_utils import align_estimates,load_model,conseguirAudioDatabase
from commons.folder_utils import generateSourcesGraph,save_sources


stft_params = nussl.STFTParams(
    window_length=config.config['STFT_WINDOW_LENGTH'],
    hop_length=config.config['STFT_HOP_LENGTH'],
    window_type=config.config['STFT_WINDOW_TYPE'],
)

def deploy(output_dir=None, audio_path=None):

    # Cargar modelo
    separator = load_model()

    # Cargar audio
    if audio_path is not None:
        audio_signal = nussl.AudioSignal(audio_path)
    else:
        test_folder = "~/.nussl/tutorial/test/"
        audio_signal = conseguirAudioDatabase(test_folder,stft_params)

    #Asignar al modelo el input
    separator.audio_signal = audio_signal

    #Recoger el output del modelo
    estimates = separator()

    #Alinear el output según el formato que queremos
    estimates = align_estimates(estimates)

    # Crear carpeta de salida
    output_dir = Path(output_dir) if output_dir else Path('../Results')
    output_dir.mkdir(parents=True, exist_ok=True)

    #Guardar fuentes en la carpeta de salida
    stems,sources_dict = save_sources(estimates,audio_signal,output_dir)

    # Generar y guardar gráfico de fuentes
    generateSourcesGraph(sources_dict, output_dir)

    # Guardar mezcla original
    audio_signal.write_audio_to_file(output_dir / 'original.wav')
    print(f"Resultados guardados en {output_dir.resolve()}")

    return stems,sources_dict
