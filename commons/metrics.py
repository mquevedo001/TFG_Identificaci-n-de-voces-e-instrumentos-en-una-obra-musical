import json
import nussl
import io
from contextlib import redirect_stdout
import torch.nn.functional as F
import torchaudio
from commons.audio_utils import spectral_convergence
from commons.model_utils import FeatureExtractor
from config import config
from geomloss import SamplesLoss  # differentiable EMD

emd_loss_fn = SamplesLoss("sinkhorn", p=1)

def evaluarFrames(frames,test_dataset,output_folder,separator):

    for i in range(frames):

        item = test_dataset[i]
        separator.audio_signal = item['mix']
        estimates = separator()

        source_keys = list(item['sources'].keys())
        estimates = {
            'vocals': estimates[0],
            'bass+drums+other': item['mix'] - estimates[0]
        }

        sources = [item['sources'][k] for k in source_keys]
        estimates = [estimates[k] for k in source_keys]

        evaluator = nussl.evaluation.BSSEvalScale(
            sources, estimates, source_labels=source_keys
        )
        scores = evaluator.evaluate()

        output_file = output_folder / (sources[0].file_name.replace('wav', 'json'))
        with open(output_file, 'w') as f:
            json.dump(scores, f, indent=4)

def reshape_to_freq(x):  # [B, F, T, S] → [B*S, F, T]
    x = x.squeeze(3)
    B, F, T, S = x.shape
    return x.permute(0, 3, 1, 2).reshape(B * S, F, T)

def reshape_to_spec(x):  # [B, F, T, S] → [B*S, 1, F, T]
    x = x.squeeze(3)
    B, F, T, S = x.shape
    return x.permute(0, 3, 1, 2).reshape(B * S, 1, F, T)


import ot
import torch
import numpy as np



def emd_loss(x, y, max_points=512):
    if x.size(0) > max_points:
        idx = torch.randperm(x.size(0))[:max_points]
        x = x[idx]
        y = y[idx]
    return emd_loss_fn(x.unsqueeze(1), y.unsqueeze(1))


def get_loss_fn(loss_type, kwargs):

    if loss_type is None:
        loss_type = config.config['MODEL_LOSS_FUNCTION']
    loss_type = loss_type.lower()


    if loss_type == 'l1':
        return lambda est, tgt: torch.mean(torch.abs(est - tgt))

    elif loss_type == 'l2':
        return lambda est, tgt: torch.mean((est - tgt) ** 2)

    elif loss_type == 'l1_freq':
        return lambda est, tgt: l1_freq(reshape_to_freq(est), reshape_to_freq(tgt))

    elif loss_type == 'l2_freq':
        return lambda est, tgt: l2_freq(reshape_to_freq(est), reshape_to_freq(tgt))

    elif loss_type == 'logl1':
        return lambda est, tgt: logl1_freq(reshape_to_freq(est), reshape_to_freq(tgt))

    elif loss_type == 'logl2':
        return lambda est, tgt: logl2_freq(reshape_to_freq(est), reshape_to_freq(tgt))

    elif loss_type == 'log_mag':
        return lambda est, tgt: log_mag_loss(reshape_to_freq(est), reshape_to_freq(tgt))

    elif loss_type == 'log_compressed_l2':
        return lambda est, tgt: log_compressed_l2(est, tgt)

    elif loss_type == 'lpsa':
        return lambda est, tgt: lpsa_loss(reshape_to_freq(est), reshape_to_freq(tgt))

    elif loss_type == 'lpsa_phase':
        mixture_phase = kwargs['mixture_phase']

        if mixture_phase is None:
            raise ValueError("Para 'lpsa_phase' necesitas pasar mixture_phase a get_loss_fn().")
        else:
            return lambda est, tgt: LPSALoss(est, tgt, mixture_phase)

    elif loss_type == 'lmrs':
        mix_mag = kwargs['mix_mag']

        if mix_mag is None:
            raise ValueError("Para 'lmrs' necesitas pasar mix_mag a get_loss_fn().")
        else:
            mixture_phase = kwargs['mixture_phase']
            mix_mag = kwargs['mix_mag']
            return lambda est, tgt: lmrs_loss(mix_mag, tgt, est)

    elif loss_type == 'mask_l1':
        return lambda est, tgt: MaskL1Loss(est, tgt)

    elif loss_type == 'l_mrs':
        return lambda est, tgt: L_MRS(
            est.sum(dim=-1),  # [B, F, T]
            tgt.sum(dim=-1)
        )

    elif loss_type == 'deep_feature':
        phi = FeatureExtractor()
        layers = ['conv1', 'conv2']
        return lambda est, tgt: deep_feature_loss(
            reshape_to_spec(est), reshape_to_spec(tgt), layers, phi
        )

    elif loss_type == 'deep_feature_emd':

        phi = FeatureExtractor()
        layers = ['conv1', 'conv2']
        return lambda est, tgt: deep_feature_loss_emd(
            reshape_to_spec(est), reshape_to_spec(tgt), layers, phi
        )
    else:
        raise ValueError(f"Función de pérdida '{loss_type}' no está implementada.")


def log_compressed_l2(pred, target, eps=1e-8):
    return torch.mean((torch.log(torch.abs(pred - target) + eps)) ** 2)

def lpsa_loss(pred_mag, true_mag, eps=1e-8):
    log_power_pred = torch.log(pred_mag ** 2 + eps)
    log_power_true = torch.log(true_mag ** 2 + eps)
    return F.mse_loss(log_power_pred, log_power_true)

def lmrs_loss(mix_mag, source_mag, estimated_mag, eps=1e-8):
    target_ratio = torch.log((source_mag + eps) / (mix_mag + eps))
    estimated_ratio = torch.log((estimated_mag + eps) / (mix_mag + eps))
    return F.mse_loss(target_ratio, estimated_ratio)

def l1_freq(mag_estimate,mag_target):
    return torch.mean(torch.abs(mag_estimate - mag_target))

def l2_freq(mag_estimate,mag_target):
    return torch.mean((mag_estimate-mag_target)**2)

def LPSALoss(estimate, target, mixture_phase):
    print("estimate shape:", estimate.shape)
    print("target shape:", target.shape)
    print("mixture_phase shape:", mixture_phase.shape)

    target_mag = torch.abs(target)
    target_phase = torch.angle(target)
    mix_phase = torch.angle(mixture_phase)

    # Si mixture_phase tiene 3 dims (B,F,T) → agregar dim para fuentes
    if mix_phase.dim() == 3:
        mix_phase = mix_phase.unsqueeze(1)  # [B,1,F,T]
    elif mix_phase.dim() == 4 and mix_phase.shape[1] == 1:
        # ya está correcto
        pass
    else:
        # print info para debug
        print("WARNING: mixture_phase unexpected shape", mix_phase.shape)

    print("target_phase shape after angle:", target_phase.shape)
    print("mix_phase shape after adjust:", mix_phase.shape)

    phase_diff = mix_phase - target_phase
    y_psa = target_mag * torch.cos(phase_diff)
    est_mag = torch.abs(estimate)

    return torch.mean((est_mag - y_psa) ** 2)




def MaskL1Loss(estimated_mask,target_sources_mag):

    mask_denominator = torch.sum(target_sources_mag, dim=1, keepdim=True) + 1e-8
    ideal_mask = target_sources_mag / mask_denominator

    loss = torch.mean(torch.abs(estimated_mask - ideal_mask))
    return loss

def MaskL1Loss(estimated_mask,target_sources_mag):

    mask_denominator = torch.sum(target_sources_mag, dim=1, keepdim=True) + 1e-8
    ideal_mask = target_sources_mag / mask_denominator

    loss = torch.mean((estimated_mask - ideal_mask) ** 2)
    return loss

def logl1_freq(y_hat,y,eps=1e-8):
    """
    LOG-L1 loss en dominio de magnitud de espectrogramas.
    y_hat, y: Tensores complejos o de magnitud (batch, freq, time, sources)
    """
    diff = torch.abs(torch.abs(y_hat) - torch.abs(y))
    sum_diff = torch.sum(diff, dim=(1, 2))  # sum over time and freq
    loss = torch.mean(torch.log10(sum_diff + eps))
    return 10 * loss

def logl2_freq(y_hat, y, eps=1e-8):
    """
    LOG-L2 loss en dominio de magnitud de espectrogramas.
    """
    diff = torch.abs(torch.abs(y_hat) - torch.abs(y)) ** 2
    sum_diff = torch.sum(diff, dim=(1, 2))
    loss = torch.mean(torch.log10(sum_diff + eps))
    return 10 * loss

def log_mag_loss(y_hat, y, eps=1e-8):
    return torch.mean(torch.abs(torch.log10(torch.abs(y_hat) + eps) - torch.log10(torch.abs(y) + eps)))

def L_MRS(y_hat,y,fft_sizes = [512,1024,2048] ,hop_sizes=[128, 256, 512], win_lengths=[512, 1024, 2048]):

    total_sc, total_mag = 0.0, 0.0
    num_resolutions = len(fft_sizes)

    for fft, hop, win in zip(fft_sizes, hop_sizes, win_lengths):
        stft = torchaudio.transforms.Spectrogram(n_fft=fft, hop_length=hop, win_length=win, power=None)
        Y_hat = stft(y_hat)
        Y = stft(y)

        total_sc += spectral_convergence(Y_hat, Y)
        total_mag += log_mag_loss(Y_hat, Y)

    return (total_sc + total_mag) / num_resolutions

def deep_feature_loss(Y_hat, Y, layers_to_use,phi = FeatureExtractor):
    """
    phi: red convolucional que extrae características (feature extractor)
    Y_hat, Y: espectrogramas estimado y real (forma: [B, 1, F, T])
    layers_to_use: lista de nombres de capas cuyas activaciones quieres usar
    """
    # Obtener activaciones intermedias
    features_hat = phi.extract_features(Y_hat, layers_to_use)
    features = phi.extract_features(Y, layers_to_use)

    loss = 0.0
    for j in layers_to_use:
        F_hat = features_hat[j]
        F_true = features[j]
        loss_j = F.mse_loss(F_hat, F_true, reduction='mean')  # Promedio sobre todos los elementos
        loss += loss_j
    return loss / len(layers_to_use)

def deep_feature_loss_emd(Y_hat, Y, layers_to_use, phi, max_points=512):
    """
    Calcula la pérdida EMD en espacio de características extraídas por phi.
    """
    features_hat = phi.extract_features(Y_hat, layers_to_use)
    features = phi.extract_features(Y, layers_to_use)

    total_loss = 0.0

    for layer in layers_to_use:
        F_hat = features_hat[layer]
        F_true = features[layer]

        # Flatten features por batch
        F_hat_flat = F_hat.view(F_hat.size(0), -1)
        F_true_flat = F_true.view(F_true.size(0), -1)

        batch_loss = 0.0
        for b in range(F_hat_flat.size(0)):
            loss_b = emd_loss(F_hat_flat[b], F_true_flat[b], max_points=max_points)
            batch_loss += loss_b

        total_loss += batch_loss / F_hat_flat.size(0)

    return total_loss / len(layers_to_use)

def run_training_and_capture_logs():
    from pipeline.train import training

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        training()  # tu función real que hace prints
    return buffer.getvalue()

