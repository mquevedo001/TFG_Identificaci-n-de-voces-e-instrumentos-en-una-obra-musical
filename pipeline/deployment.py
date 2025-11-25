import nussl
from pathlib import Path
from config import config
from commons.audio_utils import align_estimates, load_model, conseguirAudioDatabase
from commons.folder_utils import generateSourcesGraph, save_sources

import os
import torch
import numpy as np

stft_params = nussl.STFTParams(
    window_length=config.config['STFT_WINDOW_LENGTH'],
    hop_length=config.config['STFT_HOP_LENGTH'],
    window_type=config.config['STFT_WINDOW_TYPE'],
)

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
    separator = load_model()
    separator.to(config.config['DEVICE']).eval()

    print(f"[DEBUG] - Modelo : {loss_fn}+{num_sources} cargado correctamente")
    print(f"[DEBUG] - Realizando la separación con la función de pérdida: {loss_fn}")
    print(f"[DEBUG] - Separando el archivo: {audio_path}")

    # 3) Preparar input (mix_magnitude) -> OJO: stft_data es (F, T, C)
    audio_signal.stft(window_length=stft_params.window_length,
                      hop_length=stft_params.hop_length,
                      window_type=stft_params.window_type)

    mixture_stft = audio_signal.stft_data          # (F, T, C), complejo
    mixture_mag  = np.abs(mixture_stft)            # (F, T, C), real

    print("[DEBUG] mixture_stft shape (F,T,C):", mixture_stft.shape)
    print("[DEBUG] mixture_mag  shape (F,T,C):", mixture_mag.shape)

    # Ajustar canales a lo que espera el modelo
    expected_C = config.config['MODEL_NUM_CHANNELS']  # p.ej. 1
    if mixture_mag.shape[2] != expected_C:
        print(f"[WARN] canales de audio={mixture_mag.shape[2]} y el modelo espera {expected_C}. "
              f"{'Convirtiendo a mono...' if expected_C==1 else 'Recortando/ajustando canales...'}")
        if expected_C == 1:
            mixture_mag = mixture_mag.mean(axis=2, keepdims=True)   # (F, T, 1)
        else:
            mixture_mag = mixture_mag[:, :, :expected_C]             # (F, T, expected_C)

    # (F, T, C) -> (B, T, F, C) para el modelo
    mix_mag_t = torch.from_numpy(mixture_mag).float()   # (F, T, C)
    mix_mag_t = mix_mag_t.permute(1, 0, 2).contiguous() # (T, F, C)
    mix_mag_t = mix_mag_t.unsqueeze(0).to(config.config['DEVICE'])  # (1, T, F, C)

    # Comprobaciones
    B, T, F, C = mix_mag_t.shape
    nf = config.config['STFT_WINDOW_LENGTH'] // 2 + 1
    print(f"[DEBUG] mix_mag_t shape (B,T,F,C): {(B,T,F,C)} ; nf esperado={nf} ; C esperado={expected_C}")
    assert F == nf, f"Los bins de frecuencia F={F} no coinciden con nf={nf}"
    assert C == expected_C, f"Los canales C={C} no coinciden con el modelo C={expected_C}"

    batch = {'mix_magnitude': mix_mag_t.contiguous().clone()}

    # 4) Ejecuta inferencia
    with torch.backends.cudnn.flags(enabled=False):
        with torch.no_grad():
            output = separator(batch)

    # estimates_mag forma esperada por tu Embedding: (B, T, F, C, S) o (B, T, F, S) si C=1
    estimates_mag = output['estimates']
    print("[DEBUG] estimates_mag shape:", tuple(estimates_mag.shape))

    # 5) Reconstruir AudioSignal por cada stem
    # Usamos la fase de la mezcla en (F, T, C)
    phase = np.angle(mixture_stft)  # (F, T, C)
    estimates = []

    # Asegurarnos de tener eje de canales C en estimates_mag para mapear a (F,T,C)
    est_np = estimates_mag[0].detach().cpu().numpy()  # quita batch -> (T, F, C, S) ó (T, F, S) si C=1
    if est_np.ndim == 3:
        # (T, F, S) -> inserta C=1
        est_np = est_np[:, :, None, :]               # (T, F, 1, S)

    # Recorremos los stems
    for s in range(int(num_sources)):
        est_mag_tfc = est_np[..., s]        # (T, F, C)
        est_mag_ftc = np.transpose(est_mag_tfc, (1, 0, 2))  # -> (F, T, C)

        # Si C del mix es 1 pero phase tiene más (o al revés), alinea C
        if est_mag_ftc.shape[2] != phase.shape[2]:
            # Ajuste simple: si nuestro modelo es mono, replica a C canales; si es multi, recorta
            if est_mag_ftc.shape[2] == 1 and phase.shape[2] > 1:
                est_mag_ftc = np.repeat(est_mag_ftc, phase.shape[2], axis=2)
            else:
                est_mag_ftc = est_mag_ftc[:, :, :phase.shape[2]]

        est_stft = est_mag_ftc * np.exp(1j * phase)  # (F, T, C)

        est_audio = nussl.AudioSignal(stft=est_stft,
                                      sample_rate=audio_signal.sample_rate,
                                      stft_params=stft_params)
        est_audio.istft()
        estimates.append(est_audio)

    estimates = align_estimates(estimates)

    # 6) Guardar resultados
    base_out = Path('.') / 'Results' / 'Separation tests' /loss_fn / f'{num_sources}stems'

    base_out.mkdir(parents=True, exist_ok=True)

    stems, sources_dict = save_sources(estimates, audio_signal, base_out,audio_name)

    graphs_dir = '/home/martin/PycharmProjects/TFG_Identificaci-n-de-voces-e-instrumentos-en-una-obra-musical/Results/Graphs'
    generateSourcesGraph(
        sources_dict,
        output_path=graphs_dir,
        sr=44100,
        normalize_wave=False,
        db_range=(-80, 0)
    )
    audio_signal.write_audio_to_file(base_out / 'original.wav')

    print(f"Resultados guardados en {base_out.resolve()}")
    return stems, sources_dict
