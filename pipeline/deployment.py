import nussl
from pathlib import Path
from config import config
from commons.audio_utils import align_estimates, load_model, conseguirAudioDatabase, print_stem_diagnostics
from commons.folder_utils import generateSourcesGraph, save_sources
from commons.inference import run_inference

import os
import torch
import numpy as np

stft_params = nussl.STFTParams(
    window_length=config.config['STFT_WINDOW_LENGTH'],
    hop_length=config.config['STFT_HOP_LENGTH'],
    window_type=config.config['STFT_WINDOW_TYPE'],
)

device = torch.device(config.config.get(
    "DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
))


def deploy(output_dir=None, audio_path=None):
    num_sources = config.config['MODEL_NUM_SOURCES']
    loss_fn = config.config['MODEL_LOSS_FUNCTION'].lower()
    audio_name = None
    # 1) Carga del audio
    if audio_path:
        audio_signal = nussl.AudioSignal(audio_path, stft_params=stft_params)
        audio_name = os.path.basename(audio_path)
    else:
        test_folder = "~/.nussl/tutorial/test/"
        audio_signal = conseguirAudioDatabase(test_folder, stft_params)

    # 2) Cargar tu modelo (MaskInference.SeparationModel)
    print("\n[DEPLOY MODEL CONFIG]")
    print("hidden_size:", config.config['MODEL_HIDDEN_SIZE'])
    print("bidirectional:", config.config['MODEL_BIDIRECTIONAL'])
    print("num_layers:", config.config['MODEL_NUM_LAYERS'])
    separator = load_model(loss_fn=loss_fn, num_sources=num_sources)
    # 3) Ejecutar inferencia usando exactamente el mismo pipeline que evaluate_models.py
    estimates = run_inference(separator, audio_signal, num_sources)

    # 4) Guardar resultados
    base_out = Path('.') / 'Results' / 'Separation tests' / loss_fn / f'{num_sources}stems'
    base_out.mkdir(parents=True, exist_ok=True)

    stems, sources_dict = save_sources(
        estimates,
        audio_signal,
        base_out,
        audio_name,
    )
    print_stem_diagnostics(audio_signal, sources_dict)
    # 5) Guardar gráfica en una ruta válida y no hardcodeada de Linux
    graphs_dir = base_out / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    graph_name = "sources.png"
    if audio_name:
        graph_name = audio_name.replace(".wav", "_sources.png")

    generateSourcesGraph(
        sources_dict,
        output_path=graphs_dir / graph_name,
        sr=audio_signal.sample_rate,
        normalize_wave=False,
        db_range=(-80, 0),
    )

    audio_signal.write_audio_to_file(base_out / 'original.wav')

    print(f"Resultados guardados en {base_out.resolve()}")
    return stems, sources_dict
