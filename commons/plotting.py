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
            # Mover todo el batch a device y tipo float32
            batch = {k: v.to(device=device, dtype=torch.float32) if torch.is_tensor(v) else v for k, v in batch.items()}

            output = model(batch)
            estimates = output['estimates']
            references = batch['source_magnitudes']
            for est, ref in zip(estimates, references):
                est = est.squeeze()
                ref = ref.squeeze()
                sisdr = si_sdr(est, ref)  # tu función de si_sdr
                sisdrs.append(sisdr)
    return sisdrs


import torch


def si_sdr(estimation, reference, eps=1e-8):
    """
    Calcula Scale-Invariant SDR entre estimation y reference.
    Ambos deben ser tensores 1D de PyTorch.
    """
    # Normalizar señales para evitar divisiones por 0
    reference_energy = torch.sum(reference ** 2) + eps

    # Escalar referencia para mejor ajuste con estimación
    scale = torch.sum(reference * estimation) / reference_energy

    # Componente proyectada
    projection = scale * reference

    # Ruido (error)
    noise = estimation - projection

    # Calcular SI-SDR
    ratio = torch.sum(projection ** 2) / (torch.sum(noise ** 2) + eps)
    si_sdr_value = 10 * torch.log10(ratio + eps)

    return si_sdr_value.item()


def calcular_metricas_globales(model, dataloader):
    all_preds = []
    all_refs = []
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in dataloader:
            batch = prepare_batch(batch, device=device)
            output = model(batch)
            estimates = output['estimates']
            references = batch['source_magnitudes']

            all_preds.extend(estimates)
            all_refs.extend(references)

    # Calcular SI-SDR promedio usando la función si_sdr definida antes
    sisdr_values = []
    for est, ref in zip(all_preds, all_refs):
        est = est.squeeze().to(device=device, dtype=torch.float32)
        ref = ref.squeeze().to(device=device, dtype=torch.float32)
        sisdr = si_sdr(est, ref)
        sisdr_values.append(sisdr)

    si_sdr_promedio = sum(sisdr_values) / len(sisdr_values) if sisdr_values else float('nan')

    return {
        'si_sdr': si_sdr_promedio
    }

