import json
from pathlib import Path
import numpy as np
import nussl
import os

from commons.audio_utils import load_model, load_best_model
from data.test_loader import load_test_dataset
from commons.inference import run_inference

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
    "l_mrs",
    "lpsa_phase",
]

# ========================
# HELPERS
# ========================
def load_audio(path):
    return nussl.AudioSignal(str(path))

# ========================
# MAIN EVAL
# ========================
def evaluate_model(model_name, dataset):

    print(f"\n========== {model_name} ==========")

    for num_sources in [2, 4]:

        print(f"\n--- Sources: {num_sources} ---")

        ckpt_dir = CHECKPOINTS_ROOT / "Mis_modelos" / f"{model_name} checkpoints" / f"{num_sources}stems"
        best_ckpt = load_best_model(ckpt_dir)
        ckpt_path = ckpt_dir / best_ckpt

        print("Checkpoint:", ckpt_path)

        model = load_model(model_name, num_sources)

        model_results_dir = RESULTS_ROOT/'evaluate'/ model_name / f"{num_sources}stems"
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
            # EVALUACIÓN
            # ========================
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
            track_sisdr = np.mean([
                v["SI-SDR"] for v in scores.values()
            ])
            sisdr_scores.append(track_sisdr)

        # ========================
        # SUMMARY POR MODELO
        # ========================
        summary = {
            "model": model_name,
            "sources": num_sources,
            "si_sdr_mean": float(np.mean(sisdr_scores)),
            "si_sdr_median": float(np.median(sisdr_scores)),
            "si_sdr_std": float(np.std(sisdr_scores)),
            "num_tracks": len(sisdr_scores),
        }

        with open(model_results_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=4)

        print("\nSummary:", summary)


# ========================
# ENTRYPOINT
# ========================
if __name__ == "__main__":
    dataset = load_test_dataset(TEST_DATA_PATH)

    for model_name in MODELS:
        evaluate_model(model_name, dataset)