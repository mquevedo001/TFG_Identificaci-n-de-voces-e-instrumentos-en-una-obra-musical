from pathlib import Path
import csv
import json
import math

import numpy as np
import matplotlib.pyplot as plt


RESULTS_ROOT = Path("resultados_modelos/evaluate")
BASELINE_ROOT = Path("resultados_modelos/evaluate_baseline")
OUT_ROOT = Path("resultados_modelos/resumenes")

METRICS = ["SI-SDR", "SI-SAR"]


def to_float_list(value):
    if value is None:
        return []

    if isinstance(value, list):
        return [float(x) for x in value if x is not None and not math.isnan(float(x))]

    try:
        value = float(value)
        if math.isnan(value):
            return []
        return [value]
    except Exception:
        return []


def metric_values_from_track(track_json: Path, metric_name: str):
    with open(track_json, "r", encoding="utf-8") as f:
        scores = json.load(f)

    values = []

    for source_name, source_scores in scores.items():
        if source_name in ["combination", "permutation"]:
            continue

        if not isinstance(source_scores, dict):
            continue

        values.extend(to_float_list(source_scores.get(metric_name)))

    return values


def summarize_track_files(folder: Path):
    track_files = sorted(folder.glob("track_*.json"))

    summary = {
        "num_tracks": len(track_files),
    }

    for metric in METRICS:
        track_means = []

        for track_json in track_files:
            values = metric_values_from_track(track_json, metric)

            if values:
                track_means.append(float(np.mean(values)))

        key = metric.lower().replace("-", "_")

        if track_means:
            summary[f"{key}_mean"] = float(np.mean(track_means))
            summary[f"{key}_median"] = float(np.median(track_means))
            summary[f"{key}_std"] = float(np.std(track_means))
        else:
            summary[f"{key}_mean"] = None
            summary[f"{key}_median"] = None
            summary[f"{key}_std"] = None

    return summary


def load_baseline(num_sources: int):
    baseline_dir = BASELINE_ROOT / f"{num_sources}stems"

    if not baseline_dir.exists():
        return {}

    summary = summarize_track_files(baseline_dir)

    return {
        "si_sdr_mean": summary.get("si_sdr_mean"),
        "si_sar_mean": summary.get("si_sar_mean"),
    }


def collect_model_rows():
    rows = []

    if not RESULTS_ROOT.exists():
        raise FileNotFoundError(f"No existe {RESULTS_ROOT}")

    for model_dir in sorted(RESULTS_ROOT.iterdir()):
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name

        for num_sources in [2, 4]:
            stems_dir = model_dir / f"{num_sources}stems"

            if not stems_dir.exists():
                continue

            summary = summarize_track_files(stems_dir)
            baseline = load_baseline(num_sources)

            row = {
                "model": model_name,
                "sources": num_sources,
                "num_tracks": summary.get("num_tracks", 0),
                "si_sdr_mean": summary.get("si_sdr_mean"),
                "si_sdr_median": summary.get("si_sdr_median"),
                "si_sdr_std": summary.get("si_sdr_std"),
                "si_sar_mean": summary.get("si_sar_mean"),
                "si_sar_median": summary.get("si_sar_median"),
                "si_sar_std": summary.get("si_sar_std"),
                "baseline_si_sdr_mean": baseline.get("si_sdr_mean"),
                "baseline_si_sar_mean": baseline.get("si_sar_mean"),
            }

            if row["si_sdr_mean"] is not None and row["baseline_si_sdr_mean"] is not None:
                row["beats_baseline"] = row["si_sdr_mean"] > row["baseline_si_sdr_mean"]
            else:
                row["beats_baseline"] = None

            # Actualiza/crea summary.json enriquecido.
            with open(stems_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(row, f, indent=4, ensure_ascii=False)

            rows.append(row)

    return rows




def write_csv(rows, out_file: Path):
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "rank",
        "model",
        "sources",
        "num_tracks",
        "si_sdr_mean",
        "si_sdr_median",
        "si_sdr_std",
        "si_sar_mean",
        "si_sar_median",
        "si_sar_std",
        "baseline_si_sdr_mean",
        "beats_baseline",
    ]

    rows = sorted(
        rows,
        key=lambda r: -float(r["si_sdr_mean"]) if r["si_sdr_mean"] is not None else float("inf")
    )

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, row in enumerate(rows, start=1):
            row_out = dict(row)
            row_out["rank"] = i
            writer.writerow(row_out)


def make_bar_plot(rows, out_file: Path, title: str, baseline_value=None):
    out_file.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        r for r in rows
        if r["si_sdr_mean"] is not None
    ]

    rows = sorted(rows, key=lambda r: float(r["si_sdr_mean"]))

    labels = [r["model"] for r in rows]
    values = [float(r["si_sdr_mean"]) for r in rows]

    plt.figure(figsize=(9, max(4, 0.4 * len(rows))))
    plt.barh(labels, values)

    if baseline_value is not None:
        plt.axvline(float(baseline_value), linestyle="--", label="Baseline mezcla")
        plt.legend()

    plt.xlabel("SI-SDR medio (dB)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.close()


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    rows = collect_model_rows()

    for num_sources in [2, 4]:
        rows_n = [r for r in rows if int(r["sources"]) == num_sources]

        if not rows_n:
            continue

        baseline = load_baseline(num_sources)
        baseline_sisdr = baseline.get("si_sdr_mean")

        write_csv(
            rows_n,
            OUT_ROOT / f"ranking_{num_sources}stems.csv",
        )

        make_bar_plot(
            rows_n,
            OUT_ROOT / f"barplot_sisdr_{num_sources}stems.png",
            title=f"SI-SDR medio por modelo ({num_sources} stems)",
            baseline_value=baseline_sisdr,
        )

    print(f"Artefactos generados en: {OUT_ROOT}")


if __name__ == "__main__":
    main()