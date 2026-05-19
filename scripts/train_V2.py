import subprocess
import sys
from pathlib import Path

from commons.experiment_utils import MAIN_LOSSES, EXPERIMENTAL_LOSSES, ADVANCED_LOSSES


NUM_SOURCES_LIST = [2, 4]

LOSSES = MAIN_LOSSES + EXPERIMENTAL_LOSSES

MAX_EPOCHS = 100

LOG_DIR = Path("logs_train_runs_v2")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def run_training(loss_name, num_sources):
    log_file = LOG_DIR / f"{loss_name}_{num_sources}stems.log"

    cmd = [
        sys.executable,
        "main.py",
        "--mode",
        "train",
        "--lossfn",
        loss_name,
        "--numsources",
        str(num_sources),
        "--maxepochs",
        str(MAX_EPOCHS),
    ]

    print("\n" + "=" * 80)
    print("RUN:", " ".join(cmd))
    print("LOG:", log_file)
    print("=" * 80)

    with open(log_file, "w", encoding="utf-8") as f:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in process.stdout:
            print(line, end="")
            f.write(line)

        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"Falló entrenamiento loss={loss_name}, sources={num_sources}. "
            f"Revisa {log_file}"
        )


def main():
    for num_sources in NUM_SOURCES_LIST:
        for loss_name in LOSSES:
            run_training(loss_name, num_sources)


if __name__ == "__main__":
    main()