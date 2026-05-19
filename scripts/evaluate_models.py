import json
from pathlib import Path
import numpy as np
import nussl
import os

from commons.audio_utils import load_model, load_best_model
from data.test_loader import load_test_dataset
from commons.inference import run_inference
from config import config

from commons.experiment_utils import (
    MAIN_LOSSES,
    EXPERIMENTAL_LOSSES,
    resolve_checkpoint_dir,
    eval_results_dir,
    training_config_snapshot,
)


# ========================
# CONFIG
# ========================
TEST_DATA_PATH = Path("datasets/representative_test")
CHECKPOINTS_ROOT = Path("checkpoints")
RESULTS_ROOT = Path("resultados_modelos")

MODELS = [
    
    "log_compressed_l2",
    "logl2",
    "l1_freq",
    "logl1",
    "mask_l1",
    "lpsa",
    "l1",
    "l2",
    "l2_freq",
    "log_mag",
    "deep_feature",
    "deep_feature_emd",
  #  "l_mrs",
    "lpsa_phase",
]
MODELS_PROVISIONAL = [
    "logl1",
    "l1_freq",
    "lpsa",
    "l2_freq",
]

MODELS_V2 = MAIN_LOSSES + EXPERIMENTAL_LOSSES

# ========================
# HELPERS
# ========================
def ensure_mono_2d(signal):
    """
    Fuerza AudioSignal a mono con shape (1, n_samples).
    Esto evita errores de BSSEvalScale por mismatch mono/stereo.
    """
    if signal.audio_data is None:
        return signal

    if signal.audio_data.ndim == 1:
        signal.audio_data = signal.audio_data[np.newaxis, :]

    if signal.audio_data.shape[0] != 1:
        signal.to_mono(overwrite=True, keep_dims=True)

    if signal.audio_data.ndim == 1:
        signal.audio_data = signal.audio_data[np.newaxis, :]

    # Evita que queden STFTs antiguas incoherentes con audio_data.
    signal.stft_data = None
    signal.istft_data = None

    return signal


def load_audio(path):
    signal = nussl.AudioSignal(str(path))
    return ensure_mono_2d(signal)


def crop_all_to_same_length(signals):
    """
    BSSEval necesita referencias y estimaciones alineadas en longitud.
    Recortamos todas al mínimo común.
    """
    min_len = min(sig.audio_data.shape[-1] for sig in signals)

    for sig in signals:
        if sig.audio_data.ndim == 1:
            sig.audio_data = sig.audio_data[np.newaxis, :]
        sig.audio_data = sig.audio_data[:, :min_len]
        sig.stft_data = None
        sig.istft_data = None

    return min_len

# ========================
# MAIN EVAL
# ========================
def evaluate_model(model_name, dataset):

    print(f"\n========== {model_name} ==========")

    for num_sources in [2, 4]:

        print(f"\n--- Sources: {num_sources} ---")
        config.config["MODEL_NUM_SOURCES"] = int(num_sources)
        ckpt_dir = resolve_checkpoint_dir(model_name, num_sources)
        best_ckpt = load_best_model(ckpt_dir)
        ckpt_path = ckpt_dir / best_ckpt

        print("Checkpoint:", ckpt_path)

        model = load_model(model_name, num_sources)

        model_results_dir = eval_results_dir(model_name, num_sources)
        model_results_dir.mkdir(parents=True, exist_ok=True)

        sisdr_scores = []

        for i, item in enumerate(dataset):

            print(f"\nTrack {i}")

            mixture = load_audio(item['mixture'])
            sources = {k: load_audio(v) for k, v in item['sources'].items()}

            estimates = run_inference(model, mixture, num_sources)

            # ========================
            # FORMATEAR SEGÚN SOURCES
            # ========================
            if num_sources == 2:

                estimates_dict = {
                    "vocals": estimates[0],
                    "accompaniment": estimates[1]
                }

                sources_dict = {
                    "vocals": sources["vocals"],
                    "accompaniment": (
                        sources["bass"] +
                        sources["drums"] +
                        sources["other"]
                    )
                }

            else:
                keys = ['vocals', 'bass', 'drums', 'other']
                estimates_dict = dict(zip(keys, estimates))
                sources_dict = sources

            # ========================
            # EVALUACIÓN, NORMALIZAR CANALES Y LONGITUD
            # ========================

            for key in sources_dict:
                sources_dict[key] = ensure_mono_2d(sources_dict[key])

            for key in estimates_dict:
                estimates_dict[key] = ensure_mono_2d(estimates_dict[key])

            common_len = crop_all_to_same_length(
            list(sources_dict.values()) + list(estimates_dict.values())
            )

            print("[EVAL SHAPES]")
            for k, v in sources_dict.items():
                print("REF", k, v.audio_data.shape)
            for k, v in estimates_dict.items():
                print("EST", k, v.audio_data.shape)
            print("common_len:", common_len)
            
            evaluator = nussl.evaluation.BSSEvalScale(
                list(sources_dict.values()),
                list(estimates_dict.values()),
                source_labels=list(sources_dict.keys()),
            )

            scores = evaluator.evaluate()

            # ========================
            # GUARDAR JSON
            # ========================
            out_file = model_results_dir / f"track_{i}.json"
            with open(out_file, "w") as f:
                json.dump(scores, f, indent=4)

            # ========================
            # MÉTRICAS
            # ========================
            sisdr_values = []

            for source_name, source_scores in scores.items():
                if source_name in ["combination", "permutation"]:
                    continue

                if isinstance(source_scores, dict) and "SI-SDR" in source_scores:
                    sisdr = source_scores["SI-SDR"]

                    if isinstance(sisdr, list):
                        sisdr_values.extend([float(x) for x in sisdr])
                    else:
                        sisdr_values.append(float(sisdr))

            track_sisdr = np.mean(sisdr_values)
            sisdr_scores.append(track_sisdr)

        # ========================
        # SUMMARY POR MODELO
        # ========================
        baseline_4 = -4.555995227769017
        baseline_2 = -0.03977375663816929
        baseline = baseline_2 if int(num_sources) == 2 else baseline_4

        summary = {
            "model": model_name,
            "version": config.config.get("TRAIN_VERSION", "v2"),
            "sources": int(num_sources),
            "checkpoint": str(ckpt_path),
            "si_sdr_mean": float(np.mean(sisdr_scores)),
            "si_sdr_median": float(np.median(sisdr_scores)),
            "si_sdr_std": float(np.std(sisdr_scores)),
            "num_tracks": len(sisdr_scores),
            "baseline_mixture_si_sdr": float(baseline),
            "beats_mixture_baseline": bool(float(np.mean(sisdr_scores)) > baseline),
            "config": training_config_snapshot(model_name, num_sources),
        }

        with open(model_results_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=4)

        print("\nSummary:", summary)


# ========================
# ENTRYPOINT
# ========================
if __name__ == "__main__":
    dataset = load_test_dataset(TEST_DATA_PATH)

    for model_name in MODELS_V2:
        evaluate_model(model_name, dataset)