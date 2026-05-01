import json
from pathlib import Path
import numpy as np
import nussl
import os
import re

from commons.utils import load_model,load_best_model  
from evaluation.metrics import compute_sisdr  
from data.load_test_dataset import load_test_dataset

# ========================
# CONFIG
# ========================
TEST_DATA_PATH = Path("datasets/test_real")
CHECKPOINTS_ROOT = Path("checkpoints")
RESULTS_ROOT = Path("resultados_modelos")

MODELS = [
    "lpsa_phase",
    "log_compressed_l2",
    "logl2",
    "l1_freq",
    "logl1",
    "logl2",
    "mask_l1",
    "lpsa",
    "l1",
    "l2",
    "LPSA",
    "l2_freq",
    "l1_freq",
    "log_mag",
    "deep_feature",
    "deep_feature_emd",  
    "l_mrs"
    "log_compressed"
]



def load_audio(path):
    return nussl.AudioSignal(str(path))   

def evaluate_model(model_name,dataset):

    print(f"\nEvaluating: {model_name}")

    checkpoint_path = CHECKPOINTS_ROOT / "Mis_modelos" / model_name
    model_results_dir = RESULTS_ROOT / "representativo"
    sources = [2,4]
    sisdr_scores = []

    for source in sources:

        for i,item in enumerate(dataset):

            print(f"\nEvaluating: {model_name}")
            print(f"\nSources: {source}")
            print(f"\nTrack: {i}")

            checkpoint_path = os.path.join(checkpoint_path,f"{source} stems",load_best_model(checkpoint_path))
            mixture = load_audio(item['mixture'])
            sources = {k: load_audio(v) for k,v in item['sources'].items()}

            model = load_model(checkpoint_path)
            model.audio_signal = mixture
            estimates = model()

            if source == 2:
                estimates_dict = {
                    "vocals" : estimates[0],
                    "accompaniment" : estimates[1]
                }
                sources_dict = {
                    "vocals" : sources["vocals"],
                    "accompaniment" : (
                        sources["bass"] + sources["drums"] + sources["other"]
                    )
                }

            else:
                keys = ['vocals','bass','drums','other']
                estimates_dict = {k: e for k, e in zip(keys,estimates)}
                sources_dict = sources

            evaluator = nussl.evaluation.BSSEvalScale(
                list(sources_dict.values()),
                list(estimates_dict.values()),
                source_labels=list(sources_dict.keys()),
            )

            scores = evaluator.evaluate()

            out_file = model_results_dir /f"{model_name}_{source}stems" /f"track_{i}.json"
            with open(out_file, "w") as f:
                json.dump(scores,f,indent=4)
            
            track_sisdr = np.mean([
                v["SI-SDR"] for v in scores.values()
            ])

            sisdr_scores.append(track_sisdr)

            summary = {
                "si_sdr_mean": float(np.mean(sisdr_scores)),
                "si_sdr_median": float(np.median(sisdr_scores)),
                "si_sdr_std": float(np.std(sisdr_scores)),
                "num_tracks": len(sisdr_scores),
            }

            with open(model_results_dir / "summary_general_evaluation.json", "w") as f:
                json.dump(summary, f, indent=4)

            print("Summary:", summary)

            



if __name__ == "__main__":
    dataset = load_test_dataset(TEST_DATA_PATH)

    for model_name in MODELS:
        evaluate_model(model_name, dataset)