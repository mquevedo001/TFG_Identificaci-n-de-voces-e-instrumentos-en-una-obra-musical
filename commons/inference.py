import torch
import nussl
import numpy as np
from config import config

device = torch.device(config.config.get(
    "DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
))

def run_inference(model, mixture_signal, num_sources):
    model.eval()
    model.to(device)

    stft_params = nussl.STFTParams(
        window_length=int(config.config["STFT_WINDOW_LENGTH"]),
        hop_length=int(config.config["STFT_HOP_LENGTH"]),
        window_type=config.config["STFT_WINDOW_TYPE"],
    )

    mixture_signal.stft_data = None
    mixture_signal.istft_data = None
    mixture_signal.stft_params = stft_params
    mixture_signal.stft()

    mixture_stft = mixture_signal.stft_data  # [F, T, C]

    # Mismo criterio que deployment: magnitud mono a partir de canales.
    mix_mag = np.abs(mixture_stft).mean(axis=2)      # [F, T]
    mix_phase = np.angle(mixture_stft[:, :, 0])      # [F, T]

    F, T = mix_mag.shape
    expected_F = stft_params.window_length // 2 + 1
    assert F == expected_F, f"F={F}, esperado={expected_F}"

    # IMPORTANTE: no transponer aquí.
    # MaskInference.forward acepta [B, F, T] y lo convierte internamente a [B, T, F].
    mix_mag_t = torch.from_numpy(mix_mag).float().unsqueeze(0).to(device)  # [1, F, T]

    with torch.no_grad():
        output = model({"mix_magnitude": mix_mag_t})

    est = output["estimates"][0]  # [T, F, C, S] o [T, F, S]

    if est.dim() == 4:
        est = est[:, :, 0, :]  # [T, F, S]
    elif est.dim() != 3:
        raise ValueError(f"Shape inesperada en estimates: {est.shape}")

    est = est.permute(1, 0, 2).contiguous().cpu().numpy()  # [F, T, S]

    estimates = []
    for s in range(int(num_sources)):
        mag = est[:, :, s]
        complex_stft = mag * np.exp(1j * mix_phase)

        audio = nussl.AudioSignal(
            stft_data=complex_stft,
            sample_rate=mixture_signal.sample_rate,
        )
        audio.istft()
        estimates.append(audio)

    return estimates