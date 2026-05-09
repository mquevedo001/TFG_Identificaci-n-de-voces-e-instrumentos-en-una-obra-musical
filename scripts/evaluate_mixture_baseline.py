import json
from pathlib import Path
import numpy as np
import nussl

from data.test_loader import load_test_dataset

TEST_DATA_PATH = Path("datasets/representative_test")
RESULTS_ROOT = Path("resultados_modelos/evaluate_baseline")


def ensure_mono(signal):
    if signal.audio_data.ndim == 1:
        signal.audio_data = signal.audio_data[None, :]
    if signal.audio_data.shape[0] != 1:
        signal.to_mono(overwrite=True, keep_dims=True)
    return signal


def crop_all(signals):
    min_len = min(s.audio_data.shape[-1] for s in signals)
    for s in signals:
        s.audio_data = s.audio_data[:, :min_len]


def clone_audio(sig):
    out = nussl.AudioSignal(
        audio_data_array=sig.audio_data.copy(),
        sample_rate=sig.sample_rate
    )
    return out


def main():
    dataset = load_test_dataset(TEST_DATA_PATH)

    for num_sources in [2, 4]:
        out_dir = RESULTS_ROOT / f"{num_sources}stems"
        out_dir.mkdir(parents=True, exist_ok=True)

        sisdr_scores = []

        for i, item in enumerate(dataset):
            mixture = ensure_mono(nussl.AudioSignal(str(item["mixture"])))
            sources = {
                k: ensure_mono(nussl.AudioSignal(str(v)))
                for k, v in item["sources"].items()
            }

            if num_sources == 2:
                sources_dict = {
                    "vocals": sources["vocals"],
                    "accompaniment": sources["bass"] + sources["drums"] + sources["other"],
                }
                estimates_dict = {
                    "vocals": clone_audio(mixture),
                    "accompaniment": clone_audio(mixture),
                }
            else:
                keys = ["vocals", "bass", "drums", "other"]
                sources_dict = {k: sources[k] for k in keys}
                estimates_dict = {k: clone_audio(mixture) for k in keys}

            for s in list(sources_dict.values()) + list(estimates_dict.values()):
                ensure_mono(s)

            crop_all(list(sources_dict.values()) + list(estimates_dict.values()))

            evaluator = nussl.evaluation.BSSEvalScale(
                list(sources_dict.values()),
                list(estimates_dict.values()),
                source_labels=list(sources_dict.keys()),
            )

            scores = evaluator.evaluate()

            with open(out_dir / f"track_{i}.json", "w") as f:
                json.dump(scores, f, indent=4)

            track_sisdr = np.mean([
                float(scores[source]["SI-SDR"][0])
                for source in ["vocals", "accompaniment"]
                if source in scores
            ])
            sisdr_scores.append(track_sisdr)

        summary = {
            "model": "mixture_as_all_sources_baseline",
            "sources": num_sources,
            "si_sdr_mean": float(np.nanmean(sisdr_scores)),
            "si_sdr_median": float(np.nanmedian(sisdr_scores)),
            "si_sdr_std": float(np.nanstd(sisdr_scores)),
            "num_tracks": len(sisdr_scores),
        }

        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=4)

        print(summary)


if __name__ == "__main__":
    main()