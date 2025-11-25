# ==============================================================
# Script para lanzar múltiples entrenamientos de forma segura
# ==============================================================
import subprocess
import os

def main():
    loss_functions = [
        "logl1", "logl2", "log_mag",
        "log_compressed_l2", "lpsa", "mask_l1", "deep_feature", "deep_feature_emd"
    ]
    sources = ['2', '4']

    logs_folder = "logs_train_runs"
    os.makedirs(logs_folder, exist_ok=True)

    for lossfn in loss_functions:
        for num_sources in sources:
            print(f"\n=== Entrenando con lossfn={lossfn}, sources={num_sources} ===\n")

            log_prefix = f"train_{lossfn}_{num_sources}src"
            out_file = os.path.join(logs_folder, f"{log_prefix}.out")
            err_file = os.path.join(logs_folder, f"{log_prefix}.err")

            cmd = [
                "python", "main.py",
                "--mode", "train",
                "--maxepochs", "100",
                "--lossfn", lossfn,
                "--numsources", num_sources
            ]

            # 🔹 Simplificado para evitar deadlocks
            with open(out_file, "w") as f_out, open(err_file, "w") as f_err:
                result = subprocess.run(cmd, stdout=f_out, stderr=f_err)

            if result.returncode == 0:
                print(f"[OK] {lossfn} ({num_sources} fuentes) completado.\n")
            else:
                print(f"[ERROR] {lossfn} ({num_sources} fuentes) falló. Ver {err_file}\n")


if __name__ == "__main__":
    main()