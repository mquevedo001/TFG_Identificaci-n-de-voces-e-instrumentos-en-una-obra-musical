import json
import io
from contextlib import redirect_stdout

import nussl
import torch
import torch.nn.functional as F
import torchaudio

from geomloss import SamplesLoss
from commons.audio_utils import spectral_convergence,build_complex_from_mag_and_phase,reconstruct_waveforms_from_mag_phase
from commons.model_utils import FeatureExtractor
from config import config


emd_loss_fn = SamplesLoss("sinkhorn", p=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# CAMBIO: Se crea y congela una única instancia global del extractor de características.
# Antes se instanciaba dentro de get_loss_fn() en cada batch, lo que provocaba un consumo
# enorme e innecesario de memoria en deep_feature y deep_feature_emd.

#CAMBIO V2: se deja el extrator en float32 porque ahora la pipeline pide estabilidad.
feature_extractor = FeatureExtractor().to(device).float()

feature_extractor.eval()
for p in feature_extractor.parameters():
    p.requires_grad = False


def evaluarFrames(frames, test_dataset, output_folder, separator):
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


# CAMBIO: Se corrige el reshape porque el modelo trabaja con tensores [B, T, F, C, S].
# La versión anterior asumía un orden distinto y permutaba mal los ejes.
def reshape_to_freq(x):
    if x.dim() != 5:
        raise ValueError(f"reshape_to_freq esperaba un tensor 5D [B,T,F,C,S], recibió {x.shape}")
    x = x.squeeze(3)  # [B, T, F, S] si C=1
    if x.dim() != 4:
        raise ValueError(f"reshape_to_freq esperaba C=1 tras squeeze, recibió {x.shape}")
    B, T, Freq, S = x.shape
    return x.permute(0, 3, 2, 1).reshape(B * S, Freq, T)  # [B*S, F, T]


# CAMBIO: Igual que en reshape_to_freq, se corrige el orden de ejes para el formato real del modelo.
def reshape_to_spec(x):
    if x.dim() != 5:
        raise ValueError(f"reshape_to_spec esperaba un tensor 5D [B,T,F,C,S], recibió {x.shape}")
    x = x.squeeze(3)  # [B, T, F, S] si C=1
    if x.dim() != 4:
        raise ValueError(f"reshape_to_spec esperaba C=1 tras squeeze, recibió {x.shape}")
    B, T, Freq, S = x.shape
    return x.permute(0, 3, 2, 1).reshape(B * S, 1, Freq, T)  # [B*S, 1, F, T]


def extract_features(self, x, layers):
    features = {}

    x = x.to(dtype=self.conv1.weight.dtype, device=self.conv1.weight.device)

    x = self.relu(self.conv1(x))
    if "conv1" in layers:
        features["conv1"] = x.clone()

    x = self.relu(self.conv2(x))
    if "conv2" in layers:
        features["conv2"] = x.clone()

    return features


def emd_loss(x, y, max_points=512):
    if x.size(0) > max_points:
        idx = torch.randperm(x.size(0), device=x.device)[:max_points]
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
        mixture_phase = kwargs.get('mixture_phase', None)
        source_phase = kwargs.get('source_phase', None)
        if mixture_phase is None:
            raise ValueError("Para 'lpsa_phase' necesitas pasar mixture_phase a get_loss_fn().")
        # Para  PSA real hace falta la fase compleja de las fuentes objetivo, no solo su magnitud.
        return lambda est, tgt: LPSALoss(est, tgt, mixture_phase, source_phase=source_phase)

    elif loss_type == 'lmrs':
        mix_mag = kwargs.get('mix_mag', None)
        if mix_mag is None:
            raise ValueError("Para 'lmrs' necesitas pasar mix_mag a get_loss_fn().")
        return lambda est, tgt: lmrs_loss(mix_mag, tgt, est)

    elif loss_type == 'mask_l1':
        return lambda est, tgt: MaskL1Loss(est, tgt)

    elif loss_type == 'l_mrs':
        mixture_phase = kwargs.get('mixture_phase', None)
        if mixture_phase is None:
            raise ValueError("Para 'l_mrs' necesitas pasar mixture_phase a get_loss_fn().")

        return lambda est, tgt: L_MRS_from_magnitude(est, tgt, mixture_phase)

    elif loss_type == 'deep_feature':
        layers = ['conv1', 'conv2']
        return lambda est, tgt: deep_feature_loss(
            reshape_to_spec(est), reshape_to_spec(tgt), layers, feature_extractor
        )

    elif loss_type == 'deep_feature_emd':
        

        layers = ['conv1', 'conv2']  # Idealmente usar ambas capas, pero conv2 consume mucha memoria. Para pruebas, usar solo conv1.
        return lambda est, tgt: deep_feature_loss_emd(
            reshape_to_spec(est),
            reshape_to_spec(tgt),
            layers,
            feature_extractor,
            max_points=128
        )
    else:
        raise ValueError(f"Función de pérdida '{loss_type}' no está implementada.")


# CAMBIO: Se corrige la formulación. Antes se aplicaba el log al error absoluto,
# lo cual no correspondía a una pérdida log-comprimida estándar y producía escalas poco interpretables.
def log_compressed_l2(pred, target, eps=1e-8):
    pred_log = torch.log(torch.abs(pred) + eps)
    target_log = torch.log(torch.abs(target) + eps)
    return torch.mean((pred_log - target_log) ** 2)


def lpsa_loss(pred_mag, true_mag, eps=1e-8):
    log_power_pred = torch.log(pred_mag ** 2 + eps)
    log_power_true = torch.log(true_mag ** 2 + eps)
    return F.mse_loss(log_power_pred, log_power_true)


def lmrs_loss(mix_mag, source_mag, estimated_mag, eps=1e-8):
    target_ratio = torch.log((source_mag + eps) / (mix_mag + eps))
    estimated_ratio = torch.log((estimated_mag + eps) / (mix_mag + eps))
    return F.mse_loss(target_ratio, estimated_ratio)


def l1_freq(mag_estimate, mag_target):
    return torch.mean(torch.abs(mag_estimate - mag_target))


def l2_freq(mag_estimate, mag_target):
    return torch.mean((mag_estimate - mag_target) ** 2)


def LPSALoss(estimate, target, mixture_phase, source_phase):
    # estimate/target: [B, T, F, C, S]
    # mixture_phase:   [B, F, T]
    # source_phase:    puede llegar como:
    #                  [B, S, F, T]
    #                  [B, F, S, T]
    #                  [B, F, T, S]

    target_mag = torch.abs(target)
    est_mag = torch.abs(estimate)

    if mixture_phase.dim() != 3:
        raise ValueError(f"mixture_phase shape inesperada: {mixture_phase.shape}")
    if source_phase.dim() != 4:
        raise ValueError(f"source_phase shape inesperada: {source_phase.shape}")

    num_sources = target_mag.shape[-1]

    # [B, F, T] -> [B, T, F]
    mix_phase = mixture_phase.permute(0, 2, 1)

    # Detectar formato real de source_phase
    if source_phase.shape[1] == num_sources:
        # [B, S, F, T] -> [B, T, F, S]
        src_phase = source_phase.permute(0, 3, 2, 1)
    elif source_phase.shape[2] == num_sources:
        # [B, F, S, T] -> [B, T, F, S]
        src_phase = source_phase.permute(0, 3, 1, 2)
    elif source_phase.shape[3] == num_sources:
        # [B, F, T, S] -> [B, T, F, S]
        src_phase = source_phase.permute(0, 2, 1, 3)
    else:
        raise ValueError(
            f"No se puede inferir el eje de fuentes en source_phase. "
            f"shape={source_phase.shape}, num_sources={num_sources}"
        )

    # Alinear T y F
    T = min(est_mag.shape[1], target_mag.shape[1], mix_phase.shape[1], src_phase.shape[1])
    F = min(est_mag.shape[2], target_mag.shape[2], mix_phase.shape[2], src_phase.shape[2])

    est_mag = est_mag[:, :T, :F, :, :]
    target_mag = target_mag[:, :T, :F, :, :]
    mix_phase = mix_phase[:, :T, :F]
    src_phase = src_phase[:, :T, :F, :]

    # Broadcasting
    mix_phase = mix_phase.unsqueeze(-1).unsqueeze(-1)  # [B, T, F, 1, 1]
    src_phase = src_phase.unsqueeze(3)                 # [B, T, F, 1, S]

    phase_diff = mix_phase - src_phase
    y_psa = target_mag * torch.cos(phase_diff)

    if not hasattr(LPSALoss, "_debug_printed"):
        print("target_mag shape:", target_mag.shape, flush=True)
        print("mix_phase shape:", mix_phase.shape, flush=True)
        print("src_phase shape:", src_phase.shape, flush=True)
        print("phase_diff shape:", phase_diff.shape, flush=True)
        LPSALoss._debug_printed = True

    return torch.mean((est_mag - y_psa) ** 2)


def L_MRS_from_magnitude(estimate_mag, target_mag, mixture_phase):
    """
    estimate_mag:  [B, T, F, C, S]
    target_mag:    [B, T, F, C, S]
    mixture_phase: [B, F, T]
    """
    est_wave = reconstruct_waveforms_from_mag_phase(estimate_mag, mixture_phase)
    tgt_wave = reconstruct_waveforms_from_mag_phase(target_mag, mixture_phase)

    B, S, C, N = est_wave.shape

    total_loss = 0.0
    count = 0

    for b in range(B):
        for s in range(S):
            est_src = est_wave[b, s]   # [C, N]
            tgt_src = tgt_wave[b, s]   # [C, N]

            # Si C=1, pasamos a [1, N]; si no, mantenemos [C, N]
            loss_bs = L_MRS(est_src, tgt_src)
            total_loss += loss_bs
            count += 1

    return total_loss / count



def MaskL1Loss(estimated_mask, target_sources_mag):
    mask_denominator = torch.sum(target_sources_mag, dim=1, keepdim=True) + 1e-8
    ideal_mask = target_sources_mag / mask_denominator
    loss = torch.mean(torch.abs(estimated_mask - ideal_mask))
    return loss


# CAMBIO: Se conserva la variante cuadrática bajo un nombre distinto para no pisar MaskL1Loss.
def MaskL2Loss(estimated_mask, target_sources_mag):
    mask_denominator = torch.sum(target_sources_mag, dim=1, keepdim=True) + 1e-8
    ideal_mask = target_sources_mag / mask_denominator
    loss = torch.mean((estimated_mask - ideal_mask) ** 2)
    return loss


def logl1_freq(y_hat, y, eps=1e-8):
    diff = torch.abs(torch.abs(y_hat) - torch.abs(y))
    sum_diff = torch.sum(diff, dim=(1, 2))
    loss = torch.mean(torch.log10(sum_diff + eps))
    return 10 * loss


def logl2_freq(y_hat, y, eps=1e-8):
    diff = torch.abs(torch.abs(y_hat) - torch.abs(y)) ** 2
    sum_diff = torch.sum(diff, dim=(1, 2))
    loss = torch.mean(torch.log10(sum_diff + eps))
    return 10 * loss


def log_mag_loss(y_hat, y, eps=1e-8):
    return torch.mean(torch.abs(torch.log10(torch.abs(y_hat) + eps) - torch.log10(torch.abs(y) + eps)))



def L_MRS(y_hat, y, fft_sizes=[512, 1024, 2048], hop_sizes=[128, 256, 512], win_lengths=[512, 1024, 2048]):
    """
    y_hat, y: [C, N] o [N]
    """
    if y_hat.dim() == 1:
        y_hat = y_hat.unsqueeze(0)
    if y.dim() == 1:
        y = y.unsqueeze(0)

    total_sc = 0.0
    total_mag = 0.0
    num_resolutions = len(fft_sizes)

    for fft, hop, win in zip(fft_sizes, hop_sizes, win_lengths):
        stft = torchaudio.transforms.Spectrogram(
            n_fft=fft,
            hop_length=hop,
            win_length=win,
            power=None
        ).to(y_hat.device)

        Y_hat = stft(y_hat)
        Y = stft(y)

        total_sc += spectral_convergence(Y_hat, Y)
        total_mag += log_mag_loss(Y_hat, Y)

    return (total_sc + total_mag) / num_resolutions


# CAMBIO: Se evita construir gradientes para el target al extraer features.
# Antes se calculaban features de estimate y target dentro del grafo, lo que incrementaba memoria sin necesidad.
def deep_feature_loss(Y_hat, Y, layers_to_use, phi):
    dev = next(phi.parameters()).device

    Y_hat = Y_hat.to(device=dev, dtype=torch.float32)
    Y = Y.to(device=dev, dtype=torch.float32)

    with torch.amp.autocast("cuda", enabled=False):
        features_hat = phi.extract_features(Y_hat, layers_to_use)

        with torch.no_grad():
            features = phi.extract_features(Y, layers_to_use)

        loss = torch.tensor(0.0, device=dev, dtype=torch.float32)

        for j in layers_to_use:
            F_hat = features_hat[j]
            F_true = features[j]
            loss = loss + F.mse_loss(F_hat, F_true, reduction="mean")

        return loss / len(layers_to_use)


def deep_feature_loss_emd(Y_hat, Y, layers_to_use, phi, max_points=256):
    dev = next(phi.parameters()).device

    Y_hat = Y_hat.to(device=dev, dtype=torch.float32)
    Y = Y.to(device=dev, dtype=torch.float32)

    with torch.amp.autocast("cuda", enabled=False):
        features_hat = phi.extract_features(Y_hat, layers_to_use)

        with torch.no_grad():
            features = phi.extract_features(Y, layers_to_use)

        total_loss = torch.tensor(0.0, device=dev, dtype=torch.float32)

        for layer in layers_to_use:
            F_hat = features_hat[layer]
            F_true = features[layer]

            F_hat_flat = F_hat.reshape(F_hat.size(0), -1)
            F_true_flat = F_true.reshape(F_true.size(0), -1)

            batch_loss = torch.tensor(0.0, device=dev, dtype=torch.float32)

            for b in range(F_hat_flat.size(0)):
                loss_b = emd_loss(
                    F_hat_flat[b],
                    F_true_flat[b],
                    max_points=max_points,
                )
                batch_loss = batch_loss + loss_b

            total_loss = total_loss + batch_loss / F_hat_flat.size(0)

        return total_loss / len(layers_to_use)


def run_training_and_capture_logs():
    from pipeline.train import training

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        training()
    return buffer.getvalue()