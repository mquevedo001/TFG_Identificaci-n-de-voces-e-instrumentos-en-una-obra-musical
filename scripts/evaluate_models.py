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
TEST_DATA_PATH = Path(config.config["TEST_DATA_PATH"])
CHECKPOINTS_ROOT = Path("checkpoints")
RESULTS_ROOT = Path("resultados_modelos")

EVALUATION_PLAN = {
    "log_compressed_l2": (2, 4),
    "logl2": (2, 4),
    "l1_freq": (2, 4),
    "logl1": (2, 4),
    "mask_l1": (2, 4),
    "l1": (2, 4),
    "l2": (2, 4),
    "l2_freq": (2, 4),
    "log_mag": (2, 4),
    "deep_feature": (2, 4),
    "deep_feature_emd": (2, 4),
    "l_mrs": (2, 4),
    "lpsa_phase": (2, 4),

    # Solo existe evaluación final válida en 2 stems.
    "lpsa": (2,),
}

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
def evaluate_model(
    model_name,
    dataset,
    source_counts=(2, 4),
):
    """
    Evalúa un modelo para uno o varios números de fuentes.
    """

    if isinstance(
        source_counts,
        (int, np.integer),
    ):
        source_counts = (
            int(source_counts),
        )

    summaries = []

    for num_sources in source_counts:

        summary = evaluate_checkpoint(
            model_name=model_name,
            num_sources=int(num_sources),
            dataset=dataset,
        )

        summaries.append(summary)

    return summaries

def evaluate_checkpoint(
    model_name,
    num_sources,
    dataset,
    checkpoint_path=None,
    output_dir=None,
):
    """
    Evalúa un checkpoint concreto utilizando el protocolo final.

    Parameters
    ----------
    model_name : str
        Nombre de la función de pérdida/modelo.

    num_sources : int
        Número de fuentes: 2 o 4.

    dataset : list
        Dataset generado mediante load_test_dataset().

    checkpoint_path : str | Path | None
        Checkpoint concreto. Si es None, se selecciona automáticamente
        el mejor checkpoint disponible.

    output_dir : str | Path | None
        Carpeta donde guardar track_*.json y summary.json.
        Si es None, se utiliza la carpeta oficial de evaluate_v2.
    """

    model_name = str(model_name).lower()
    num_sources = int(num_sources)

    if num_sources not in (2, 4):
        raise ValueError(
            f"num_sources debe ser 2 o 4, recibido {num_sources}"
        )

    print(f"\n========== {model_name} ==========")
    print(f"\n--- Sources: {num_sources} ---")

    config.config["MODEL_NUM_SOURCES"] = num_sources
    config.config["MODEL_LOSS_FUNCTION"] = model_name

    # =========================================================
    # CHECKPOINT
    # =========================================================

    if checkpoint_path is None:
        ckpt_dir = resolve_checkpoint_dir(
            model_name,
            num_sources,
        )

        best_ckpt = load_best_model(ckpt_dir)
        ckpt_path = ckpt_dir / best_ckpt

    else:
        ckpt_path = Path(checkpoint_path)

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No existe el checkpoint: {ckpt_path}"
        )

    print("Checkpoint:", ckpt_path)

    model = load_model(
        model_name,
        num_sources,
        checkpoint_path=ckpt_path,
    )

    # =========================================================
    # OUTPUT
    # =========================================================

    if output_dir is None:
        model_results_dir = eval_results_dir(
            model_name,
            num_sources,
        )
    else:
        model_results_dir = Path(output_dir)

    model_results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =========================================================
    # ACUMULADORES
    # =========================================================

    sisdr_scores = []
    sisar_scores = []
    sisdri_scores = []
    mix_sisdr_scores = []

    # =========================================================
    # TRACKS
    # =========================================================

    for i, item in enumerate(dataset):

        print(f"\nTrack {i}")

        mixture = load_audio(item["mixture"])

        sources = {
            key: load_audio(path)
            for key, path in item["sources"].items()
        }

        estimates = run_inference(
            model,
            mixture,
            num_sources,
        )

        # -----------------------------------------------------
        # ORGANIZAR FUENTES
        # -----------------------------------------------------

        if num_sources == 2:

            estimates_dict = {
                "vocals": estimates[0],
                "accompaniment": estimates[1],
            }

            sources_dict = {
                "vocals": sources["vocals"],
                "accompaniment": (
                    sources["bass"]
                    + sources["drums"]
                    + sources["other"]
                ),
            }

        else:

            source_order = [
                "vocals",
                "bass",
                "drums",
                "other",
            ]

            estimates_dict = dict(
                zip(source_order, estimates)
            )

            sources_dict = {
                key: sources[key]
                for key in source_order
            }

        # -----------------------------------------------------
        # MONO + ALINEACIÓN
        # -----------------------------------------------------

        for key in sources_dict:
            sources_dict[key] = ensure_mono_2d(
                sources_dict[key]
            )

        for key in estimates_dict:
            estimates_dict[key] = ensure_mono_2d(
                estimates_dict[key]
            )

        common_len = crop_all_to_same_length(
            list(sources_dict.values())
            + list(estimates_dict.values())
        )

        print("[EVAL SHAPES]")

        for key, value in sources_dict.items():
            print(
                "REF",
                key,
                value.audio_data.shape,
            )

        for key, value in estimates_dict.items():
            print(
                "EST",
                key,
                value.audio_data.shape,
            )

        print("common_len:", common_len)

        # -----------------------------------------------------
        # BSSEval
        # -----------------------------------------------------

        evaluator = nussl.evaluation.BSSEvalScale(
            list(sources_dict.values()),
            list(estimates_dict.values()),
            source_labels=list(
                sources_dict.keys()
            ),
        )

        scores = evaluator.evaluate()

        # -----------------------------------------------------
        # RESULTADO DE LA PISTA
        # -----------------------------------------------------

        out_file = (
            model_results_dir
            / f"track_{i}.json"
        )

        with open(
            out_file,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                scores,
                f,
                indent=4,
                ensure_ascii=False,
            )

        # -----------------------------------------------------
        # MÉTRICAS
        # -----------------------------------------------------

        sisdr_values = metric_values_from_scores(
            scores,
            "SI-SDR",
        )

        sisar_values = metric_values_from_scores(
            scores,
            "SI-SAR",
        )

        sisdri_values = metric_values_from_scores(
            scores,
            "SI-SDRi",
        )

        mix_sisdr_values = metric_values_from_scores(
            scores,
            "MIX-SI-SDR",
        )

        track_sisdr = mean_or_none(
            sisdr_values
        )

        track_sisar = mean_or_none(
            sisar_values
        )

        track_sisdri = mean_or_none(
            sisdri_values
        )

        track_mix_sisdr = mean_or_none(
            mix_sisdr_values
        )

        if track_sisdr is not None:
            sisdr_scores.append(
                track_sisdr
            )

        if track_sisar is not None:
            sisar_scores.append(
                track_sisar
            )

        if track_sisdri is not None:
            sisdri_scores.append(
                track_sisdri
            )

        if track_mix_sisdr is not None:
            mix_sisdr_scores.append(
                track_mix_sisdr
            )

    # =========================================================
    # SUMMARY
    # =========================================================

    baseline_summary = load_baseline_summary(
        num_sources
    )

    baseline_sisdr = None
    baseline_sisar = None

    if baseline_summary is not None:
        baseline_sisdr = baseline_summary.get(
            "si_sdr_mean"
        )

        baseline_sisar = baseline_summary.get(
            "si_sar_mean"
        )

    model_sisdr_mean = mean_or_none(
        sisdr_scores
    )

    model_sisar_mean = mean_or_none(
        sisar_scores
    )

    model_sisdri_mean = mean_or_none(
        sisdri_scores
    )

    model_mix_sisdr_mean = mean_or_none(
        mix_sisdr_scores
    )

    summary = {
        "model": model_name,
        "version": config.config.get(
            "TRAIN_VERSION",
            "v2",
        ),
        "sources": num_sources,
        "checkpoint": str(ckpt_path),

        "si_sdr_mean": model_sisdr_mean,
        "si_sdr_median": median_or_none(
            sisdr_scores
        ),
        "si_sdr_std": std_or_none(
            sisdr_scores
        ),

        "si_sar_mean": model_sisar_mean,
        "si_sar_median": median_or_none(
            sisar_scores
        ),
        "si_sar_std": std_or_none(
            sisar_scores
        ),

        "si_sdr_i_mean": model_sisdri_mean,
        "si_sdr_i_median": median_or_none(
            sisdri_scores
        ),
        "si_sdr_i_std": std_or_none(
            sisdri_scores
        ),

        "mix_si_sdr_mean_from_tracks":
            model_mix_sisdr_mean,

        "num_tracks": len(
            sisdr_scores
        ),

        "baseline_mixture_si_sdr":
            baseline_sisdr,

        "baseline_mixture_si_sar":
            baseline_sisar,

        "beats_mixture_baseline": (
            bool(
                model_sisdr_mean
                > baseline_sisdr
            )
            if (
                model_sisdr_mean is not None
                and baseline_sisdr is not None
            )
            else None
        ),

        "beats_mixture_baseline_by_sisdri": (
            bool(model_sisdri_mean > 0)
            if model_sisdri_mean is not None
            else None
        ),

        "config": training_config_snapshot(
            model_name,
            num_sources,
        ),
    }

    summary_path = (
        model_results_dir
        / "summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print("\nSummary:", summary)
    print(
        f"Resultados guardados en: "
        f"{model_results_dir.resolve()}"
    )

    return summary
    
# ========================
# ENTRYPOINT
# ========================
if __name__ == "__main__":
    dataset = load_test_dataset(TEST_DATA_PATH)

    for model_name,source_counts in EVALUATION_PLAN.items():evaluate_model(model_name, dataset,source_counts=source_counts)
        