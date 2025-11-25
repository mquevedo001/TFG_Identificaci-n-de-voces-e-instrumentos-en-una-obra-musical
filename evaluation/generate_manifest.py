#!/usr/bin/env python3
# make_manifest_musdb.py
#
# Genera un manifest_2stems CSV para test a partir de carpetas de stems:
#   sources_root/
#       vocals/
#       drums/
#       bass/
#       other/
#
# Para 4 fuentes: mixture = vocals + drums + bass + other
# Para 2 fuentes: mixture = vocals + drums + bass + other  (y source_1 = accompaniment = drums + bass + other)
#
# CSV resultante:
#   - 4 fuentes: mixture_path,source_0_path,source_1_path,source_2_path,source_3_path  (vocals,drums,bass,other)
#   - 2 fuentes: mixture_path,source_0_path,source_1_path                              (vocals,accompaniment)
#
# Requisitos: numpy, soundfile

import argparse
import csv
from pathlib import Path
from typing import List, Dict, Set
import numpy as np
import soundfile as sf

STEMS = ["vocals", "drums", "bass", "other"]

def list_basenames(folder: Path, exts: List[str]) -> Set[str]:
    names = set()
    if not folder.exists():
        return names
    for ext in exts:
        for p in folder.glob(f"*{ext}"):
            names.add(p.stem)
    return names

def load_audio(path: Path):
    # Devuelve (audio np.float32, sr), audio shape = (num_frames, num_channels)
    audio, sr = sf.read(path, always_2d=True, dtype="float32")
    return audio, sr

def peak_normalize(x: np.ndarray, peak: float = 0.999) -> np.ndarray:
    m = np.max(np.abs(x))
    if m > peak and m > 0:
        x = x * (peak / m)
    return x

def ensure_same_sr_ch(arrays: List[np.ndarray], srs: List[int], names: List[str]) -> bool:
    if len(set(srs)) != 1:
        print(f"[WARN] SR distintos para {names}: {srs}. Se omite esta pista.")
        return False
    chs = [a.shape[1] for a in arrays]
    if len(set(chs)) != 1:
        print(f"[WARN] Nº de canales distintos para {names}: {chs}. Se omite esta pista.")
        return False
    return True

def truncate_to_min_len(arrays: List[np.ndarray]) -> List[np.ndarray]:
    min_len = min(a.shape[0] for a in arrays)
    return [a[:min_len] for a in arrays]

def main():
    ap = argparse.ArgumentParser(description="Genera manifest_2stems MUSDB-like sumando stems para 2 o 4 fuentes.")
    ap.add_argument("--sources_root", required=True, type=str,
                    help="Carpeta con subcarpetas vocals/drums/bass/other.")
    ap.add_argument("--out_root", required=True, type=str,
                    help="Carpeta donde se guardarán los WAV generados (mixes y, si procede, accompaniment).")
    ap.add_argument("--out_csv", required=True, type=str,
                    help="Ruta del CSV de manifest_2stems a generar.")
    ap.add_argument("--num_sources", required=True, type=int, choices=[2, 4],
                    help="2 = vocals+accompaniment, 4 = vocals,drums,bass,other.")
    ap.add_argument("--exts", type=str, default=".wav",
                    help="Extensiones a considerar separadas por comas (p.ej. .wav,.flac). Por defecto: .wav")
    ap.add_argument("--no_normalize", action="store_true",
                    help="Por defecto se evita clipping con peak-normalize. Usa este flag para desactivar.")
    args = ap.parse_args()

    sources_root = Path(args.sources_root).resolve()
    out_root = Path(args.out_root).resolve()
    out_csv = Path(args.out_csv).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    mixes_dir = out_root / "mixes"
    mixes_dir.mkdir(parents=True, exist_ok=True)
    acc_dir = out_root / "accompaniment"
    if args.num_sources == 2:
        acc_dir.mkdir(parents=True, exist_ok=True)

    exts = [e.strip() for e in args.exts.split(",") if e.strip()]
    # Intersección de basenames presentes en todas las subcarpetas requeridas
    name_sets = []
    for stem in STEMS:
        stem_dir = sources_root / stem
        s = list_basenames(stem_dir, exts)
        if not s:
            print(f"[ERROR] No se encontraron archivos en {stem_dir} con extensiones {exts}.")
            return
        name_sets.append(s)

    common_names = sorted(set.intersection(*map(set, name_sets)))
    if not common_names:
        print("[ERROR] No hay basenames comunes en todas las subcarpetas de stems.")
        return

    rows: List[List[str]] = []
    skipped = 0

    for name in common_names:
        # Rutas de stems
        paths: Dict[str, Path] = {}
        ok = True
        for stem in STEMS:
            found = None
            for ext in exts:
                p = (sources_root / stem / f"{name}{ext}")
                if p.exists():
                    found = p
                    break
            if found is None:
                ok = False
                print(f"[WARN] Falta {stem} para '{name}'. Se omite.")
                break
            paths[stem] = found
        if not ok:
            skipped += 1
            continue

        # Cargar stems
        audios = []
        srs = []
        for stem in STEMS:
            a, sr = load_audio(paths[stem])
            audios.append(a)
            srs.append(sr)

        if not ensure_same_sr_ch(audios, srs, [f"{name}/{s}" for s in STEMS]):
            skipped += 1
            continue

        # Alinear por truncado
        audios = truncate_to_min_len(audios)  # [vocals, drums, bass, other]
        sr = srs[0]

        # mixture = suma de todos
        mix = audios[0] + audios[1] + audios[2] + audios[3]
        if not args.no_normalize:
            mix = peak_normalize(mix)

        mix_path = mixes_dir / f"{name}_mixture.wav"
        sf.write(mix_path, mix, sr, subtype="PCM_16")

        if args.num_sources == 4:
            # manifest_2stems 4 fuentes: mix, vocals, drums, bass, other
            row = [
                str(mix_path),
                str(paths["vocals"]),
                str(paths["drums"]),
                str(paths["bass"]),
                str(paths["other"]),
            ]
        else:
            # 2 fuentes: vocals + accompaniment (drums+bass+other)
            acc = audios[1] + audios[2] + audios[3]
            if not args.no_normalize:
                acc = peak_normalize(acc)
            acc_path = acc_dir / f"{name}_accompaniment.wav"
            sf.write(acc_path, acc, sr, subtype="PCM_16")
            row = [
                str(mix_path),
                str(paths["vocals"]),
                str(acc_path),
            ]
        rows.append(row)

    # Escribir CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.num_sources == 4:
        header = ["mixture_path"] + [f"source_{i}_path" for i in range(4)]
    else:
        header = ["mixture_path", "source_0_path", "source_1_path"]  # 0=vocals, 1=accompaniment

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    print(f"[OK] Manifest guardado en: {out_csv}")
    print(f"[INFO] Mezclas renderizadas en: {mixes_dir}")
    if args.num_sources == 2:
        print(f"[INFO] Acompaniments renderizados en: {acc_dir}")
    print(f"[INFO] Total pistas: {len(rows)} | Omitidas: {skipped}")

if __name__ == "__main__":
    main()

