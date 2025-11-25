import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from commons.model_utils import prepare_batch

def guardar_resultados(model_name, metrics, sisdr_list, loss_history, base_path='resultados_modelos'):
    model_dir = Path(base_path) / 'individuales' / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    with open(model_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    # Pérdidas por iteración
    if isinstance(loss_history, dict) and 'iter' in loss_history:
        plt.figure()
        plt.plot(loss_history['iter'])
        plt.xlabel('Iteración')
        plt.ylabel('Pérdida')
        plt.title(f'Curva de pérdida (iter) - {model_name}')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(model_dir / 'curva_loss_iter.png')
        plt.close()

    # Pérdidas por época
    if isinstance(loss_history, dict) and 'epoch' in loss_history:
        plt.figure()
        plt.plot(loss_history['epoch'])
        plt.xlabel('Época')
        plt.ylabel('Pérdida promedio')
        plt.title(f'Curva de pérdida (epoch) - {model_name}')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(model_dir / 'curva_loss_epoch.png')
        plt.close()

    # Boxplot SI-SDR
    plt.figure()
    plt.boxplot(sisdr_list)
    plt.ylabel('SI-SDR (dB)')
    plt.title(f'Distribución SI-SDR - {model_name}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(model_dir / 'boxplot_sisdr.png')
    plt.close()

    np.save(model_dir / 'sisdr_values.npy', np.array(sisdr_list))

def generar_comparativas(base_path='resultados_modelos'):
    """
    Lee los metrics.json de todos los modelos y genera gráficos comparativos
    """
    indiv_path = Path(base_path) / 'individuales'
    comp_path = Path(base_path) / 'comparativas'
    comp_path.mkdir(parents=True, exist_ok=True)

    data = []
    for model_dir in indiv_path.iterdir():
        metrics_file = model_dir / 'metrics.json'
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
                metrics['modelo'] = model_dir.name
                data.append(metrics)

    df = pd.DataFrame(data)
    df.set_index('modelo', inplace=True)

    # Barplot de SI-SDR
    if 'si_sdr' in df.columns:
        plt.figure()
        df['si_sdr'].sort_values().plot(kind='barh', title='Comparación SI-SDR')
        plt.xlabel('SI-SDR (dB)')
        plt.tight_layout()
        plt.savefig(comp_path / 'sisdr_barplot.png')
        plt.close()

    # Gráfico de tiempos de inferencia (si existe)
    if 'inference_time' in df.columns:
        plt.figure()
        df['inference_time'].sort_values().plot(kind='barh', title='Tiempo de Inferencia (s)')
        plt.xlabel('Tiempo (s)')
        plt.tight_layout()
        plt.savefig(comp_path / 'inference_times.png')
        plt.close()

    # Radar chart para algunas métricas comunes
    radar_metrics = ['si_sdr', 'f1_score', 'precision', 'recall', 'accuracy']
    df_radar = df[[m for m in radar_metrics if m in df.columns]]
    if df_radar.shape[1] >= 3:
        from math import pi
        labels = df_radar.columns.tolist()
        num_vars = len(labels)

        for idx, (model_name, row) in enumerate(df_radar.iterrows()):
            angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
            angles += angles[:1]
            values = row.tolist()
            values += values[:1]

            plt.figure()
            ax = plt.subplot(111, polar=True)
            plt.xticks(angles[:-1], labels)
            ax.plot(angles, values, linewidth=2, label=model_name)
            ax.fill(angles, values, alpha=0.25)
            plt.title(f'Radar Chart - {model_name}')
            plt.tight_layout()
            plt.savefig(comp_path / f'radar_{model_name}.png')
            plt.close()


import torchaudio

import nussl


def calcular_sisdr_por_muestra(model, dataloader):
    sisdrs = []
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: (v.to(device=device, dtype=torch.float32) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}

            output = model(batch)
            # shapes esperadas: (B, T, F, C, S)
            estimates = output['estimates']
            references = batch['source_magnitudes']

            # Validación de forma
            if estimates.shape[:-1] != references.shape[:-1]:
                raise RuntimeError(f"Shapes incompatibles (sin eje S): "
                                   f"{estimates.shape[:-1]} vs {references.shape[:-1]}")
            if estimates.shape[-1] != references.shape[-1]:
                raise RuntimeError(f"Número de fuentes distinto entre pred y ref: "
                                   f"{estimates.shape[-1]} vs {references.shape[-1]}")

            B, T, F, C, S = estimates.shape

            # Reordena a (B, S, -1) y calcula SI-SDR por (batch, fuente)
            est_flat = estimates.permute(0, 4, 1, 2, 3).reshape(B, S, -1)  # (B,S,N)
            ref_flat = references.permute(0, 4, 1, 2, 3).reshape(B, S, -1)  # (B,S,N)

            for b in range(B):
                for s in range(S):
                    sisdr = si_sdr(est_flat[b, s], ref_flat[b, s])  # usa tu si_sdr con tensores 1D
                    sisdrs.append(sisdr)
    return sisdrs



import torch


def si_sdr(estimation, reference, eps=1e-8):
    estimation = estimation.float()
    reference = reference.float()

    reference_energy = torch.sum(reference ** 2) + eps
    scale = torch.sum(reference * estimation) / reference_energy
    projection = scale * reference
    noise = estimation - projection
    ratio = torch.sum(projection ** 2) / (torch.sum(noise ** 2) + eps)
    return (10 * torch.log10(ratio + eps)).item()



def calcular_metricas_globales(model, dataloader):
    model.eval()
    device = next(model.parameters()).device
    sisdr_values = []

    with torch.no_grad():
        for batch in dataloader:
            batch = prepare_batch(batch, device=device)
            output = model(batch)
            estimates = output['estimates']          # (B,T,F,C,S)
            references = batch['source_magnitudes']  # (B,T,F,C,S)

            if estimates.shape != references.shape:
                raise RuntimeError(f"pred y ref deben tener misma shape; got {estimates.shape} vs {references.shape}")

            B, T, F, C, S = estimates.shape
            est_flat = estimates.permute(0, 4, 1, 2, 3).reshape(B, S, -1)
            ref_flat = references.permute(0, 4, 1, 2, 3).reshape(B, S, -1)

            for b in range(B):
                for s in range(S):
                    sisdr_values.append(si_sdr(est_flat[b, s], ref_flat[b, s]))

    si_sdr_promedio = float(np.mean(sisdr_values)) if sisdr_values else float('nan')
    return {'si_sdr': si_sdr_promedio}


