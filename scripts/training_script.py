import os
import subprocess
import torch
from datetime import datetime

def run_and_tee(cmd, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Line-buffering: importante para ver salida en vivo
    # universal_newlines=True => texto
    with open(out_path, "w", encoding="utf-8") as f_out:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,   # <-- CLAVE: un solo stream (evita bloqueos)
            text=True,
            bufsize=1,                 # line-buffered
            universal_newlines=True
        )

        for line in proc.stdout:
            print(line, end="")        # salida a terminal
            f_out.write(line)          # salida a fichero

        return proc.wait()

def main():
    #FINISHED TRAININGS ->          
    #"lpsa_phase",
    #"log_compressed_l2",
    # "logl2",
    #"l1_freq",
    #"logl1",
    #"logl2",
    #"log_mag",
    #"mask_l1",
    #"lpsa",
    #"l1",
    #"l2",
    #"LPSA",
    #"l2_freq",
    #"l1_freq",
    #"log_mag",
    #"LPSA",
    #         "deep_feature",
    #    "deep_feature_emd",  
    #"log_compressed_l2",
    #PENDING TRAININGS ->
    #"deep_feature_emd",
    #"l_mrs"
    #"lpsa_phase"
    #"log_compressed_l2"


    loss_functions = [

 #       "l_mrs",
        "lpsa_phase",
        

    ]
    sources = ["4"]

    logs_folder = "logs_train_runs"

    for lossfn in loss_functions:
        for num_sources in sources:
            torch.cuda.empty_cache()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(logs_folder, f"{ts}_train_{lossfn}_{num_sources}src.log")

            cmd = [
                "python", "main.py",
                "--mode", "train",
                "--maxepochs", "100",
                "--lossfn", lossfn,
                "--numsources", num_sources
            ]

            print(f"\n=== RUN: lossfn={lossfn} | sources={num_sources} | log={log_file} ===\n")
            code = run_and_tee(cmd, log_file)

            if code == 0:
                print(f"\n[OK] Finalizado: {lossfn} ({num_sources} fuentes)\n")
            else:
                print(f"\n[ERROR] Código {code}: {lossfn} ({num_sources} fuentes). Mira el log.\n")

if __name__ == "__main__":
    main()