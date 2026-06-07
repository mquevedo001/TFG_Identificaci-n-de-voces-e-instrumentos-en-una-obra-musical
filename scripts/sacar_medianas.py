import numpy as np
from pathlib import Path
import os

base_dir = Path(__file__).parent
base_dir = Path(str(base_dir).replace("scripts",""))

PATH_MODELOS = Path("resultados_modelos/individuales")
NOMBRE_FILE_SISDR_VALUES = "sisdr_values.npy"

RESULTS_DIR = Path(base_dir / PATH_MODELOS)
OUTPUT_FILE_MEDIAN = Path(RESULTS_DIR / "medianas_sisdr_values.txt")
OUTPUT_FILE_STD = Path(RESULTS_DIR / "std_sisdr_values.txt")
OUTPUT_FILE_MEAN = Path(RESULTS_DIR / "mean_sisdr_values.txt")

values_mean = {}
values_median = {}
values_std = {}

for folder_model in os.listdir(RESULTS_DIR):

    folder_model_path = Path( base_dir / PATH_MODELOS / folder_model)   
    if folder_model_path.is_dir():
        for file in os.listdir(folder_model_path):   
            if file.endswith(NOMBRE_FILE_SISDR_VALUES):    
                p = Path( folder_model_path / file)            
                
                value = np.load(p)

                value_std = np.std(value)
                value_mean = np.mean(value)
                value_median = np.median(value)

                values_median[folder_model] = value_median
                values_std[folder_model] = value_std
                values_mean[folder_model] = value_mean


for key, value in values_median.items():

    with open(OUTPUT_FILE_MEDIAN, "a") as f:
        f.write(f"{key}: {value}\n")

for key, value in values_std.items():

    with open(OUTPUT_FILE_STD, "a") as f:
        f.write(f"{key}: {value}\n")

for key, value in values_mean.items():

    with open(OUTPUT_FILE_MEAN, "a") as f:
        f.write(f"{key}: {value}\n")
            
