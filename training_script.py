import subprocess
import os
from config import config

def main():
    # Lista de las funciones de pérdida que soporta get_loss_fn, en minúsculas
    loss_functions = [
        "l1", "l2", "l1_freq", "l2_freq", "logl1", "logl2", "log_mag",
        "log_compressed_l2", "lpsa", "mask_l1","deep_feature", "deep_feature_emd"
    ]

    # lpsa_phase , lmrs , l_mrs

    logs_folder = "logs_train_runs"
    os.makedirs(logs_folder, exist_ok=True)

    for lossfn in loss_functions:
        print(f"Ejecutando entrenamiento con lossfn = {lossfn} ...")

        config.config['BATCH_SIZE'] = 1
        # Construir comando
        cmd = \
            [
            "python", "/home/martin/PycharmProjects/TFG/main.py",
            "--mode", "train",
            "--maxepochs", "100",
            "--lossfn", lossfn,
            "--numsources","2"
        ]

        try:
            # Ejecutar comando y capturar salida y error
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            # Guardar stdout y stderr en archivos
            with open(os.path.join(logs_folder, f"train_{lossfn}.out"), "w") as f_out:
                f_out.write(result.stdout)
            with open(os.path.join(logs_folder, f"train_{lossfn}.err"), "w") as f_err:
                f_err.write(result.stderr)

            print(f"Terminó correctamente con lossfn={lossfn}")

        except subprocess.CalledProcessError as e:
            print(f"Error ejecutando lossfn={lossfn}. Ver logs para detalles.")
            with open(os.path.join(logs_folder, f"train_{lossfn}.out"), "w") as f_out:
                f_out.write(e.stdout or "")
            with open(os.path.join(logs_folder, f"train_{lossfn}.err"), "w") as f_err:
                f_err.write(e.stderr or "")

if __name__ == "__main__":
    main()
