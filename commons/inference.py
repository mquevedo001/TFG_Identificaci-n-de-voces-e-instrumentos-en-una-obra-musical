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
        window_length=512,
        hop_length=128,
        window_type="sqrt_hann",
    )

    mixture_signal.stft_data = None
    mixture_signal.istft_data = None
    mixture_signal.stft_params = stft_params
    mixture_signal.stft()

    mixture_stft = mixture_signal.stft_data  # (F, T, C)

    mix_mag = np.abs(mixture_stft)[:, :, 0]   # (F, T)
    mix_phase = np.angle(mixture_stft)[:, :, 0]

    F, T = mix_mag.shape
    expected_F = stft_params.window_length // 2 + 1
    assert F == expected_F

    print(f"[DEBUG] STFT: F={F}, T={T}")

    # ========================
    #  INPUT CORRECTO
    # ========================
    mix_mag = torch.from_numpy(mix_mag).float()  # (F, T)
    mix_mag = mix_mag.T                          # (T, F)
    mix_mag = mix_mag.unsqueeze(0).to(device)    # (1, T, F)

    print("[DEBUG] MODEL INPUT:", mix_mag.shape)

    # ========================
    # FORWARD
    # ========================
    print("MINI CHECK:")
    print("F:", F)
    print("T:", T)
    print("MODEL EXPECTED INPUT_SIZE =", model.layers.model.recurrent_stack.rnn.input_size)

    print("\n===== MODEL EXPECTATION CHECK =====")

    for name, p in model.named_parameters():
        if "rnn" in name:
            print(name, p.shape)
    print("\n===== INPUT TRACE =====")
    print("mix_mag:", mix_mag.shape)
    with torch.no_grad():
        output = model({"mix_magnitude": mix_mag})

    # ========================
    # OUTPUT
    # ========================
    est = output["estimates"] if isinstance(output, dict) else output
    est = est.squeeze(0)

    # esperado: (F, T, S)
    if est.shape[0] != F:
        est = est.permute(1, 0, 2)

    est = est.cpu().numpy()

    estimates = []

    for s in range(num_sources):
        mag = est[:, :, s]  # (F, T)

        complex_stft = mag * np.exp(1j * mix_phase)

        audio = nussl.AudioSignal(
            stft_data=complex_stft,
            sample_rate=mixture_signal.sample_rate
        )
        audio.istft()

        estimates.append(audio)

    return estimates