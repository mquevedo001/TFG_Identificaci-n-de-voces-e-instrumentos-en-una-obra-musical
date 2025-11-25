# ==============================================================
# evaluate_models.py
# Recorre modelos entrenados (por loss y nº de fuentes), carga
# checkpoints y evalúa en un conjunto de TEST.
# ==============================================================

import os
import sys
import re
import csv
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import torch
import nussl

# === Tu modelo y utilidades ===
from models.Mi_modelo.mask_inference import MaskInference
from commons.model_utils import prepare_batch
from commons.plotting import (
    guardar_resultados, calcular_metricas_globales, calcular_sisdr_por_muestra
)
from data.data_loader import get_data  # se usa como "plan B" si no hay loader de test
from config import config


# ==============================================================
# 1) VARIABLES DE RUTAS DE TEST (déjalas vacías y rellena la que prefieras)
#    Usa SOLO una de estas tres opciones.
# ==============================================================

# (OPCIÓN 1) Carpeta con mezclas (mixtures) y subcarpetas con stems por archivo.
#   Estructura esperada:
#     TEST_MIX_DIR/
#       track_0001/
#         mixture.wav
#         sources/
#           source_00.wav
#           source_01.wav
#           ... (hasta num_sources - 1)
TEST_MIX_DIR = ""  # e.g., "/ruta/a/test_set"

# (OPCIÓN 2) Manifest de test (CSV/TSV; delimitador auto-detectado aunque no tenga extensión)
#   Columnas: mixture_path, source_0_path, source_1_path, ...
TEST_MANIFEST = "/home/martin/PycharmProjects/TFG_Identificaci-n-de-voces-e-instrumentos-en-una-obra-musical/evaluation/manifest_2stems"

# (OPCIÓN 3) Lista de ejemplos en memoria (paths absolutos)
#   Ejemplo:
#   TEST_ITEMS = [
#       {"mixture": "/abs/mixture1.wav", "sources": ["/abs/s0.wav", "/abs/s1.wav", ...]},
#       {"mixture": "/abs/mixture2.wav", "sources": [ ... ]},
#   ]
TEST_ITEMS: List[Dict[str, List[str]]] = []


# ==============================================================
# 2) CONFIGURACIÓN DEL *SWEEP* A EVALUAR
# ==============================================================

LOSS_FUNCTIONS = [
    "logl1", "logl2", "log_mag",
    "log_compressed_l2", "lpsa", "mask_l1", "deep_feature", "deep_feature_emd"
]
NUM_SOURCES_LIST = ["2", "4"]  # coincide con tu lanzador de entrenamiento

BATCH_SIZE = 8
NUM_WORKERS = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# carpeta raíz donde guardaste checkpoints durante el entrenamiento
# (el entrenamiento creaba: checkpoints/Mis_modelos/{loss} checkpoints/{N}stems/)
CKPT_ROOT = Path("checkpoints") / "Mis_modelos"

# Dónde guardar la salida de evaluación (métricas, curvas, etc.)
EVAL_OUT_ROOT = Path("results_eval")
EVAL_OUT_ROOT.mkdir(parents=True, exist_ok=True)

# Dónde guardan tus utilidades los resultados agregados
RESULTADOS_BASE = "resultados_modelos"  # usado por guardar_resultados()
INDIV_OUT = Path(RESULTADOS_BASE) / "individuales"
INDIV_OUT.mkdir(parents=True, exist_ok=True)


# ==============================================================
# 3) DATASET/LOADER DE TEST
# ==============================================================

import numpy as np  # asegúrate de tener este import arriba

class TestSeparationDataset(torch.utils.data.Dataset):
    """
    Dataset simple a partir de (mixture_path, [source_paths...]).
    Devuelve tensores compatibles con `prepare_batch`:
      - mixture_magnitude: (T, F, C)  float32
      - mixture_phase:     (T, F, C)  float32   (radianes)
      - source_magnitudes: (T, F, C, S) float32
    """
    def __init__(self, items: List[Dict[str, List[str]]], stft_params: nussl.STFTParams):
        super().__init__()
        self.items = list(items)
        self.stft_params = stft_params

    def __len__(self):
        return len(self.items)

    def _stft(self, sig: nussl.AudioSignal):
        # Usa los campos de STFTParams (en tu versión no existe stft_params=...)
        sig.stft(
            window_length=self.stft_params.window_length,
            hop_length=self.stft_params.hop_length,
            window_type=self.stft_params.window_type,
        )

    @staticmethod
    def _to_TFC_from_stft(stft_complex: np.ndarray, num_channels: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Acepta stft_complex con forma (?,?,?) en algún orden de (F, T, C) o (C, F, T) o (F, C, T),
        y devuelve (mag, phase) en (T, F, C).
        """
        if stft_complex.ndim != 3:
            raise ValueError(f"STFT con dims inesperadas: {stft_complex.shape}")

        F0, F1, F2 = stft_complex.shape
        # Detecta dónde está C
        if F0 == num_channels:           # (C, F, T) -> (T, F, C)
            order = (2, 1, 0)
        elif F2 == num_channels:         # (F, T, C) -> (T, F, C)
            order = (1, 0, 2)
        elif F1 == num_channels:         # (F, C, T) -> (T, F, C)
            order = (2, 0, 1)
        else:
            raise ValueError(f"No encuentro eje de canales {num_channels} en {stft_complex.shape}")

        mag = np.abs(stft_complex).transpose(order)    # (T, F, C)
        phase = np.angle(stft_complex).transpose(order)  # (T, F, C)
        return mag, phase

    def __getitem__(self, idx: int):
        ex = self.items[idx]

        # --- Mezcla ---
        mix = nussl.AudioSignal(ex["mixture"])
        self._stft(mix)
        mix_mag_np, mix_phase_np = self._to_TFC_from_stft(mix.stft_data, mix.num_channels)

        # --- Fuentes ---
        src_mags_np = []
        for p in ex["sources"]:
            s = nussl.AudioSignal(p)
            self._stft(s)
            s_mag_np, _ = self._to_TFC_from_stft(s.stft_data, s.num_channels)
            src_mags_np.append(s_mag_np)

        # (Opcional) Alinear longitudes temporales por seguridad
        T_min = min([mix_mag_np.shape[0]] + [sm.shape[0] for sm in src_mags_np])
        if mix_mag_np.shape[0] != T_min:
            mix_mag_np = mix_mag_np[:T_min]
            mix_phase_np = mix_phase_np[:T_min]
        src_mags_np = [sm[:T_min] for sm in src_mags_np]

        # A tensores
        mix_mag = torch.from_numpy(mix_mag_np.copy()).float()
        mix_phase = torch.from_numpy(mix_phase_np.copy()).float()
        source_magnitudes = torch.stack([torch.from_numpy(sm.copy()).float() for sm in src_mags_np], dim=-1)  # (T, F, C, S)

        return {
            "mixture_magnitude": mix_mag,
            "mixture_phase": mix_phase,
            "source_magnitudes": source_magnitudes,
        }


def _discover_from_folder(root: str, num_sources: int) -> List[Dict[str, List[str]]]:
    """Busca pares (mixture, sources[]) en una carpeta con estructura estándar."""
    items = []
    root = Path(root)
    if not root.exists():
        return items

    for track_dir in sorted(root.iterdir()):
        if not track_dir.is_dir():
            continue
        mix_candidates = list(track_dir.glob("mixture.*"))
        if not mix_candidates:
            mix_candidates = list(track_dir.glob("*.wav"))
        if not mix_candidates:
            continue
        mixture = str(mix_candidates[0])

        src_dir = track_dir / "sources"
        if not src_dir.exists():
            continue
        srcs = sorted([str(p) for p in src_dir.glob("*.wav")])
        if len(srcs) < num_sources:
            continue
        items.append({"mixture": mixture, "sources": srcs[:num_sources]})
    return items


def _read_manifest(manifest_path: str, num_sources: int) -> List[Dict[str, List[str]]]:
    """Lee manifest con delimitador auto-detectado; si hay más fuentes que `num_sources`, recorta."""
    items = []
    if not manifest_path or not Path(manifest_path).exists():
        return items

    with open(manifest_path, "r", encoding="utf-8", newline="") as f:
        # sniff delimitador
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
            delim = dialect.delimiter
        except Exception:
            # heurística
            delim = "," if "," in sample.splitlines()[0] else "\t"

        reader = csv.DictReader(f, delimiter=delim)
        # columnas esperadas: mixture_path, source_0_path, source_1_path, ...
        for row in reader:
            mix = row.get("mixture_path", "").strip()
            srcs = []
            for i in range(num_sources):
                key = f"source_{i}_path"
                srcs.append(row.get(key, "").strip())
            if mix and all(srcs):
                items.append({"mixture": mix, "sources": srcs})
    return items


def build_test_loader(stft_params, num_sources: int):
    # 1) TEST_ITEMS
    if TEST_ITEMS:
        ds = TestSeparationDataset(TEST_ITEMS, stft_params)
        return torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True), "items"

    # 2) TEST_MANIFEST
    if TEST_MANIFEST:
        items = _read_manifest(TEST_MANIFEST, num_sources)
        if items:
            ds = TestSeparationDataset(items, stft_params)
            return torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True), "manifest"

    # 3) TEST_MIX_DIR
    if TEST_MIX_DIR:
        items = _discover_from_folder(TEST_MIX_DIR, num_sources)
        if items:
            ds = TestSeparationDataset(items, stft_params)
            return torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True), "folder"

    # 4) Plan B: usar val_dataloader y forzar coherencia de nº fuentes
    prev_nsrc = config.config.get('MODEL_NUM_SOURCES', None)
    try:
        config.config['MODEL_NUM_SOURCES'] = num_sources
        train_data, val_data = get_data(
            stft_params,
            config.config['MAX_MIXTURES'],
            config.config['COHERENT_PROB']
        )
    finally:
        if prev_nsrc is not None:
            config.config['MODEL_NUM_SOURCES'] = prev_nsrc

    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True
    )
    return val_loader, "val_fallback"


# ==============================================================
# 4) CARGA DE CHECKPOINTS
# ==============================================================

def find_best_checkpoint(dir_path: Path) -> Optional[Path]:
    """Prioriza archivos con 'best' y .pt/.pth; si no, el más reciente."""
    if not dir_path.exists():
        return None
    cands = list(dir_path.glob("*.pt")) + list(dir_path.glob("*.pth"))
    if not cands:
        cands = list((dir_path / "checkpoints").glob("*.pt")) + list((dir_path / "checkpoints").glob("*.pth"))
    if not cands:
        return None
    best_like = [p for p in cands if re.search(r"best", p.name, re.IGNORECASE)]
    if best_like:
        return sorted(best_like, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    return sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def safe_load_model(model: nussl.ml.SeparationModel, ckpt_path: Path):
    """Carga pesos soportando varios formatos comunes de checkpoint."""
    obj = torch.load(str(ckpt_path), weights_only=False, map_location="cpu")
    if isinstance(obj, dict):
        for key in ["model", "state_dict", "model_state_dict", "net"]:
            if key in obj and isinstance(obj[key], dict):
                model.load_state_dict(obj[key], strict=False)
                return
        try:
            model.load_state_dict(obj, strict=False)
            return
        except Exception:
            pass
    raise RuntimeError(f"No se pudo interpretar el checkpoint: {ckpt_path}")


# ==============================================================
# 5) MÉTRICAS AUXILIARES (CSV + resumen + recopilador final)
# ==============================================================

def save_sisdr_csv(sisdr_list, model_tag, out_dir=INDIV_OUT):
    out = Path(out_dir) / model_tag
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"si_sdr": np.array(sisdr_list, dtype=float)})
    df.to_csv(out / "sisdr_per_sample.csv", index=False)

    mean = float(df.si_sdr.mean())
    med  = float(df.si_sdr.median())
    std  = float(df.si_sdr.std(ddof=1))
    n    = int(len(df))
    ci95 = float(1.96 * std / np.sqrt(max(n, 1)))
    summary = {"mean": mean, "median": med, "std": std, "n": n, "ci95": ci95}
    with open(out / "sisdr_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] CSV y resumen SI-SDR en {out}")

def collect_summaries(base=INDIV_OUT):
    rows=[]
    base = Path(base)
    if not base.exists():
        print("[AVISO] No hay directorio de resultados para recopilar.")
        return None
    for d in base.iterdir():
        if not d.is_dir():
            continue
        sfile = d / "sisdr_summary.json"
        mfile = d / "metrics.json"  # por si dejaste más cosas ahí
        rec = {"modelo": d.name}
        if sfile.exists():
            with open(sfile) as f: rec.update(json.load(f))
        if mfile.exists():
            with open(mfile) as f:
                try:
                    rec.update(json.load(f))
                except Exception:
                    pass
        rows.append(rec)
    if not rows:
        print("[AVISO] No se encontraron resúmenes.")
        return None
    df = pd.DataFrame(rows).set_index("modelo")
    df = df.sort_values("mean", ascending=False)
    print("\n=== Ranking por media de SI-SDR (↑) ===")
    print(df[["mean","median","std","ci95"]].round(3))
    out = Path(RESULTADOS_BASE) / "comparativas"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "ranking_sisdr.csv")
    print(f"[OK] Ranking guardado en {out / 'ranking_sisdr.csv'}")
    return df


# ==============================================================
# 6) EVALUACIÓN
# ==============================================================

@torch.no_grad()
def evaluate_model_on_loader(
    model: nussl.ml.SeparationModel,
    data_loader: torch.utils.data.DataLoader
):
    model.eval()
    # Nota: tus funciones ya hacen forward y comparan contra GT por DataLoader.
    sisdr_por_muestra = calcular_sisdr_por_muestra(model, data_loader)
    metricas_dict = calcular_metricas_globales(model, data_loader)
    return sisdr_por_muestra, metricas_dict


def build_model_for_eval(n_stems: int) -> nussl.ml.SeparationModel:
    # Mantiene coherencia con tu script de entrenamiento.
    stft_params = nussl.STFTParams(
        window_length=config.config['STFT_WINDOW_LENGTH'],
        hop_length=config.config['STFT_HOP_LENGTH'],
        window_type=config.config['STFT_WINDOW_TYPE'],
    )
    nf = stft_params.window_length // 2 + 1

    model = MaskInference.build(
        nf,
        num_audio_channels=config.config['MODEL_NUM_CHANNELS'],
        hidden_size=config.config['MODEL_HIDDEN_SIZE'],
        num_layers=config.config['MODEL_NUM_LAYERS'],
        bidirectional=config.config['MODEL_BIDIRECTIONAL'],
        dropout=config.config['MODEL_DROPOUT'],
        num_sources=n_stems,
        activation=config.config['MODEL_ACTIVATION']
    )
    return model


def main():
    # STFT params para construir el DataLoader de test
    stft_params = nussl.STFTParams(
        window_length=config.config['STFT_WINDOW_LENGTH'],
        hop_length=config.config['STFT_HOP_LENGTH'],
        window_type=config.config['STFT_WINDOW_TYPE'],
    )

    for lossfn in LOSS_FUNCTIONS:
        for num_sources in NUM_SOURCES_LIST:
            nsrc = int(num_sources)
            print(f"\n=== Evaluando lossfn={lossfn} | sources={nsrc} ===")

            # Ruta donde se guardó el entrenamiento correspondiente
            ckpt_dir = CKPT_ROOT / f"{lossfn} checkpoints" / f"{nsrc}stems"
            ckpt_path = find_best_checkpoint(ckpt_dir)
            if ckpt_path is None:
                print(f"[AVISO] No se encontró checkpoint en: {ckpt_dir}")
                continue

            print(f"Usando checkpoint: {ckpt_path}")

            # Construye DataLoader de TEST
            test_loader, mode = build_test_loader(stft_params, nsrc)
            print(f"Conjunto de test cargado vía: {mode} | #batches ≈ {len(test_loader)}")

            # Construye y carga modelo
            model = build_model_for_eval(nsrc).to(DEVICE)
            try:
                safe_load_model(model, ckpt_path)
            except Exception as e:
                print(f"[ERROR] Cargando checkpoint: {e}")
                continue

            # Evalúa
            sisdr_list, metrics = evaluate_model_on_loader(model, test_loader)

            # Guarda resultados (reuse de tus utilidades)
            eval_tag = f"{lossfn}_{nsrc}stems"
            guardar_resultados(
                model_name=f"{eval_tag}_EVAL",
                metrics=metrics,
                sisdr_list=sisdr_list,
                loss_history={"iter": [], "epoch": []},
            )

            # CSV + resumen rápido por modelo
            save_sisdr_csv(sisdr_list, model_tag=f"{eval_tag}_EVAL")

            # Además, guarda un resumen legible en disco
            out_dir = EVAL_OUT_ROOT / f"{lossfn}" / f"{nsrc}stems"
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "metrics.txt", "w", encoding="utf-8") as f:
                f.write(f"Checkpoint: {ckpt_path}\n")
                f.write(f"Test loader mode: {mode}\n")
                f.write("=== MÉTRICAS GLOBALES ===\n")
                for k, v in metrics.items():
                    f.write(f"{k}: {v}\n")
                f.write("\n=== SI-SDR por muestra ===\n")
                for i, s in enumerate(sisdr_list):
                    f.write(f"{i}\t{s}\n")

            print(f"[OK] {eval_tag} → métricas guardadas en {out_dir}\n")

    # Al final: ranking global
    collect_summaries(INDIV_OUT)


if __name__ == "__main__":
    main()
