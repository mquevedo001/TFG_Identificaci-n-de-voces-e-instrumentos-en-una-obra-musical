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
#    "lpsa",
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

MODELS_V2 = MODELS 

# ========================
# HELPERS
# ========================

def metric_values_from_scores(scores, metric_name):
    values = []

    for source_name, source_scores in scores.items():
        if source_name in ["combination", "permutation"]:
            continue

        if not isinstance(source_scores, dict):
            continue

        if metric_name not in source_scores:
            continue

        value = source_scores[metric_name]

        if isinstance(value, list):
            values.extend([float(x) for x in value])
        else:
            values.append(float(value))

    return values


def mean_or_none(values):
    if not values:
        return None

    return float(np.mean(values))


def median_or_none(values):
    if not values:
        return None

    return float(np.median(values))


def std_or_none(values):
    if not values:
        return None

    return float(np.std(values))


def load_baseline_summary(num_sources):
    baseline_path = (
        Path(config.config.get("RESULTS_ROOT", "resultados_modelos"))
        / "evaluate_baseline"
        / f"{int(num_sources)}stems"
        / "summary.json"
    )

    if not baseline_path.exists():
        return None

    with open(baseline_path, "r", encoding="utf-8") as f:
        return json.load(f)
    

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
def evaluate_model(model_name, dataset,source_counts=(2,4)):


    print(f"\n========== {model_name} ==========")

    for num_sources in source_counts:

        print(f"\n--- Sources: {num_sources} ---")
        config.config["MODEL_NUM_SOURCES"] = int(num_sources)
        ckpt_dir = resolve_checkpoint_dir(model_name, num_sources)
        best_ckpt = load_best_model(ckpt_dir)
        ckpt_path = ckpt_dir / best_ckpt

        print("Checkpoint:", ckpt_path)

        model = load_model(model_name, num_sources,checkpoint_path=ckpt_path)

        model_results_dir = eval_results_dir(model_name, num_sources)
        model_results_dir.mkdir(parents=True, exist_ok=True)

        sisdr_scores = []
        sisar_scores = []
        sisdri_scores = []
        mix_sisdr_scores = []
        
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
            sisdr_values = metric_values_from_scores(scores, "SI-SDR")
            sisar_values = metric_values_from_scores(scores, "SI-SAR")
            sisdri_values = metric_values_from_scores(scores, "SI-SDRi")
            mix_sisdr_values = metric_values_from_scores(scores, "MIX-SI-SDR")

            track_sisdr = mean_or_none(sisdr_values)
            track_sisar = mean_or_none(sisar_values)
            track_sisdri = mean_or_none(sisdri_values)
            track_mix_sisdr = mean_or_none(mix_sisdr_values)

            if track_sisdr is not None:
                sisdr_scores.append(track_sisdr)

            if track_sisar is not None:
                sisar_scores.append(track_sisar)

            if track_sisdri is not None:
                sisdri_scores.append(track_sisdri)

            if track_mix_sisdr is not None:
                mix_sisdr_scores.append(track_mix_sisdr)

        # ========================
        # SUMMARY POR MODELO
        # ========================
        baseline_summary = load_baseline_summary(num_sources)

        baseline_sisdr = None
        baseline_sisar = None

        if baseline_summary is not None:
            baseline_sisdr = baseline_summary.get("si_sdr_mean")
            baseline_sisar = baseline_summary.get("si_sar_mean")

        model_sisdr_mean = mean_or_none(sisdr_scores)
        model_sisar_mean = mean_or_none(sisar_scores)
        model_sisdri_mean = mean_or_none(sisdri_scores)
        model_mix_sisdr_mean = mean_or_none(mix_sisdr_scores)

        summary = {
            "model": model_name,
            "version": config.config.get("TRAIN_VERSION", "v2"),
            "sources": int(num_sources),
            "checkpoint": str(ckpt_path),

            "si_sdr_mean": model_sisdr_mean,
            "si_sdr_median": median_or_none(sisdr_scores),
            "si_sdr_std": std_or_none(sisdr_scores),

            "si_sar_mean": model_sisar_mean,
            "si_sar_median": median_or_none(sisar_scores),
            "si_sar_std": std_or_none(sisar_scores),

            "si_sdr_i_mean": model_sisdri_mean,
            "si_sdr_i_median": median_or_none(sisdri_scores),
            "si_sdr_i_std": std_or_none(sisdri_scores),

            "mix_si_sdr_mean_from_tracks": model_mix_sisdr_mean,

            "num_tracks": len(sisdr_scores),

            "baseline_mixture_si_sdr": baseline_sisdr,
            "baseline_mixture_si_sar": baseline_sisar,

            "beats_mixture_baseline": (
                bool(model_sisdr_mean > baseline_sisdr)
                if model_sisdr_mean is not None and baseline_sisdr is not None
                else None
            ),

            "beats_mixture_baseline_by_sisdri": (
                bool(model_sisdri_mean > 0)
                if model_sisdri_mean is not None
                else None
            ),

            "config": training_config_snapshot(model_name, num_sources),
        }

        with open(model_results_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=4)

        print("\nSummary:", summary)

def evaluate_checkpoint(
    model_name,
    num_sources,
    dataset,
    checkpoint_path=None,
    output_dir=None,
):
    print(f"\n========== {model_name} ==========")

    print(f"\n--- Sources: {num_sources} ---")
    config.config["MODEL_NUM_SOURCES"] = int(num_sources)
    ckpt_dir = resolve_checkpoint_dir(model_name, num_sources)
    best_ckpt = load_best_model(ckpt_dir)
    ckpt_path = ckpt_dir / best_ckpt

    print("Checkpoint:", ckpt_path)

    model = load_model(model_name, num_sources,checkpoint_path=ckpt_path)

    model_results_dir = eval_results_dir(model_name, num_sources)
    model_results_dir.mkdir(parents=True, exist_ok=True)

    sisdr_scores = []
    sisar_scores = []
    sisdri_scores = []
    mix_sisdr_scores = []
    
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
        sisdr_values = metric_values_from_scores(scores, "SI-SDR")
        sisar_values = metric_values_from_scores(scores, "SI-SAR")
        sisdri_values = metric_values_from_scores(scores, "SI-SDRi")
        mix_sisdr_values = metric_values_from_scores(scores, "MIX-SI-SDR")

        track_sisdr = mean_or_none(sisdr_values)
        track_sisar = mean_or_none(sisar_values)
        track_sisdri = mean_or_none(sisdri_values)
        track_mix_sisdr = mean_or_none(mix_sisdr_values)

        if track_sisdr is not None:
            sisdr_scores.append(track_sisdr)

        if track_sisar is not None:
            sisar_scores.append(track_sisar)

        if track_sisdri is not None:
            sisdri_scores.append(track_sisdri)

        if track_mix_sisdr is not None:
            mix_sisdr_scores.append(track_mix_sisdr)

        # ========================
        # SUMMARY POR MODELO
        # ========================
        baseline_summary = load_baseline_summary(num_sources)

        baseline_sisdr = None
        baseline_sisar = None

        if baseline_summary is not None:
            baseline_sisdr = baseline_summary.get("si_sdr_mean")
            baseline_sisar = baseline_summary.get("si_sar_mean")

        model_sisdr_mean = mean_or_none(sisdr_scores)
        model_sisar_mean = mean_or_none(sisar_scores)
        model_sisdri_mean = mean_or_none(sisdri_scores)
        model_mix_sisdr_mean = mean_or_none(mix_sisdr_scores)

        summary = {
            "model": model_name,
            "version": config.config.get("TRAIN_VERSION", "v2"),
            "sources": int(num_sources),
            "checkpoint": str(ckpt_path),

            "si_sdr_mean": model_sisdr_mean,
            "si_sdr_median": median_or_none(sisdr_scores),
            "si_sdr_std": std_or_none(sisdr_scores),

            "si_sar_mean": model_sisar_mean,
            "si_sar_median": median_or_none(sisar_scores),
            "si_sar_std": std_or_none(sisar_scores),

            "si_sdr_i_mean": model_sisdri_mean,
            "si_sdr_i_median": median_or_none(sisdri_scores),
            "si_sdr_i_std": std_or_none(sisdri_scores),

            "mix_si_sdr_mean_from_tracks": model_mix_sisdr_mean,

            "num_tracks": len(sisdr_scores),

            "baseline_mixture_si_sdr": baseline_sisdr,
            "baseline_mixture_si_sar": baseline_sisar,

            "beats_mixture_baseline": (
                bool(model_sisdr_mean > baseline_sisdr)
                if model_sisdr_mean is not None and baseline_sisdr is not None
                else None
            ),

            "beats_mixture_baseline_by_sisdri": (
                bool(model_sisdri_mean > 0)
                if model_sisdri_mean is not None
                else None
            ),

            "config": training_config_snapshot(model_name, num_sources),
        }

        with open(model_results_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=4)

        print("\nSummary:", summary)    
        print(f"\n========== {model_name} ==========")
        print(f"\n--- Sources: {num_sources} ---")
        config.config["MODEL_NUM_SOURCES"] = int(num_sources)
        ckpt_dir = resolve_checkpoint_dir(model_name, num_sources)
        best_ckpt = load_best_model(ckpt_dir)
        ckpt_path = ckpt_dir / best_ckpt

        print("Checkpoint:", ckpt_path)

        model = load_model(model_name, num_sources,checkpoint_path=ckpt_path)

        model_results_dir = eval_results_dir(model_name, num_sources)
        model_results_dir.mkdir(parents=True, exist_ok=True)

        sisdr_scores = []
        sisar_scores = []
        sisdri_scores = []
        mix_sisdr_scores = []
        
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
            sisdr_values = metric_values_from_scores(scores, "SI-SDR")
            sisar_values = metric_values_from_scores(scores, "SI-SAR")
            sisdri_values = metric_values_from_scores(scores, "SI-SDRi")
            mix_sisdr_values = metric_values_from_scores(scores, "MIX-SI-SDR")

            track_sisdr = mean_or_none(sisdr_values)
            track_sisar = mean_or_none(sisar_values)
            track_sisdri = mean_or_none(sisdri_values)
            track_mix_sisdr = mean_or_none(mix_sisdr_values)

            if track_sisdr is not None:
                sisdr_scores.append(track_sisdr)

            if track_sisar is not None:
                sisar_scores.append(track_sisar)

            if track_sisdri is not None:
                sisdri_scores.append(track_sisdri)

            if track_mix_sisdr is not None:
                mix_sisdr_scores.append(track_mix_sisdr)

        # ========================
        # SUMMARY POR MODELO
        # ========================
        baseline_summary = load_baseline_summary(num_sources)

        baseline_sisdr = None
        baseline_sisar = None

        if baseline_summary is not None:
            baseline_sisdr = baseline_summary.get("si_sdr_mean")
            baseline_sisar = baseline_summary.get("si_sar_mean")

        model_sisdr_mean = mean_or_none(sisdr_scores)
        model_sisar_mean = mean_or_none(sisar_scores)
        model_sisdri_mean = mean_or_none(sisdri_scores)
        model_mix_sisdr_mean = mean_or_none(mix_sisdr_scores)

        summary = {
            "model": model_name,
            "version": config.config.get("TRAIN_VERSION", "v2"),
            "sources": int(num_sources),
            "checkpoint": str(ckpt_path),

            "si_sdr_mean": model_sisdr_mean,
            "si_sdr_median": median_or_none(sisdr_scores),
            "si_sdr_std": std_or_none(sisdr_scores),

            "si_sar_mean": model_sisar_mean,
            "si_sar_median": median_or_none(sisar_scores),
            "si_sar_std": std_or_none(sisar_scores),

            "si_sdr_i_mean": model_sisdri_mean,
            "si_sdr_i_median": median_or_none(sisdri_scores),
            "si_sdr_i_std": std_or_none(sisdri_scores),

            "mix_si_sdr_mean_from_tracks": model_mix_sisdr_mean,

            "num_tracks": len(sisdr_scores),

            "baseline_mixture_si_sdr": baseline_sisdr,
            "baseline_mixture_si_sar": baseline_sisar,

            "beats_mixture_baseline": (
                bool(model_sisdr_mean > baseline_sisdr)
                if model_sisdr_mean is not None and baseline_sisdr is not None
                else None
            ),

            "beats_mixture_baseline_by_sisdri": (
                bool(model_sisdri_mean > 0)
                if model_sisdri_mean is not None
                else None
            ),

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