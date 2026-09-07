# train.py
import math

import torch
import nussl

from models.Mi_modelo.mask_inference import MaskInference
from data.data_loader import get_data
from config import config
from commons.metrics import get_loss_fn
from commons.plotting import guardar_resultados, calcular_metricas_globales, calcular_sisdr_por_muestra
from commons.model_utils import prepare_batch

from ignite.handlers import EarlyStopping
from ignite.engine import Events
from ignite.metrics import Average
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine

from commons.experiment_utils import (
    checkpoint_dir,
    get_loss_group,
    training_config_snapshot,
    save_json,
)

# ============================
# HELPERS
# ============================
def choose_batch_settings(loss_type, num_stems):

    loss_type = str(loss_type).lower()
    group = get_loss_group(loss_type)

    if group == "experimental":

        actual_batch_size = int(config.config.get("BATCH_SIZE_EXPERIMENTAL", 4))
        effective_batch_size = int(config.config.get("EFFECTIVE_BATCH_SIZE_EXPERIMENTAL", 32))

    else:
        if int(num_stems) == 4: actual_batch_size = int(config.config.get("BATCH_SIZE_4STEMS_MAIN", 24))
        else: actual_batch_size = int(config.config.get("BATCH_SIZE_2STEMS_MAIN", 32))
    
        effective_batch_size = int(config.config.get("EFFECTIVE_BATCH_SIZE_MAIN", 96))

    accumulation_steps = math.ceil(effective_batch_size / actual_batch_size)

    return actual_batch_size, accumulation_steps, effective_batch_size


def get_mix_magnitude_btf(batch, targets):
    """
    Devuelve mix_magnitude en formato [B, T, F].
    targets esperado: [B, T, F, C, S]
    """
    mix = batch.get("mix_magnitude", None)

    if mix is None: mix = batch.get("mixture_magnitude", None)
    if mix is None: raise ValueError("No encuentro 'mix_magnitude' ni 'mixture_magnitude' en el batch.")
    if mix.dim() == 4 and mix.shape[-1] == 1: mix = mix.squeeze(-1)
    if mix.dim() != 3: raise ValueError(f"mix_magnitude debería ser 3D, recibido {mix.shape}")

    
    target_f = targets.shape[2]

    if mix.shape[1] == target_f: mix = mix.permute(0, 2, 1).contiguous() # [B, F, T] -> [B, T, F]
    elif mix.shape[2] == target_f: mix = mix.contiguous() # [B, T, F]
    else:raise ValueError(f"No puedo alinear mix_magnitude={mix.shape} con targets={targets.shape}")
        
    return mix


def ideal_amplitude_mask_loss(pred_mask, batch, targets, eps=1e-8):
    """
    pred_mask: [B, T, F, C, S]
    targets:   [B, T, F, C, S]
    """
    mix_btf = get_mix_magnitude_btf(batch, targets)
    mix = mix_btf.unsqueeze(3).unsqueeze(-1)

    ideal_mask = targets / (mix + eps)
    ideal_mask = torch.clamp(ideal_mask, 0.0, 1.0)

    return torch.mean(torch.abs(pred_mask - ideal_mask))


def compute_training_loss(loss_type, output, batch, kwargs):

    estimates = output["estimates"]
    targets = batch["source_magnitudes"]

    if str(loss_type).lower() == "mask_l1": return ideal_amplitude_mask_loss(output["mask"], batch, targets)
        
    loss_fn = get_loss_fn(loss_type, kwargs)
    return loss_fn(estimates, targets)

def build_loss_kwargs(loss_type, batch):
    loss_type = str(loss_type).lower()
    kwargs = {}

    if loss_type == 'lpsa_phase' and ('mixture_phase' not in batch or 'source_phase' not in batch): raise ValueError("El batch no contiene 'mixture_phase' o 'source_phase'.")
    if loss_type in ['lmrs','l_mrs'] and 'mixture_magnitude' not in batch: raise ValueError("El batch no contiene 'mixture_magnitude'.")
    if loss_type == 'l_mrs' and 'mixture_phase' not in batch: raise ValueError('El batch no contiene mixture_phase.')

    if loss_type == 'lpsa_phase':
        kwargs['mixture_phase'] = batch['mixture_phase']
        kwargs['source_phase'] = batch['source_phase']

    if loss_type in ['lmrs', 'l_mrs']: kwargs['mix_mag'] = batch['mixture_magnitude'] 
    if loss_type == 'l_mrs':kwargs['mixture_phase'] = batch['mixture_phase']

    return kwargs
    
# ============================
# TRAINING
# ============================
def training():
    # ----------------------------
    # Config / device
    # ----------------------------
    loss_fn_name        = config.config['MODEL_LOSS_FUNCTION']
    num_stems           = int(config.config['MODEL_NUM_SOURCES'])
    device              = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ----------------------------
    # STFT
    # ----------------------------
    stft_params = nussl.STFTParams(
        window_length   = int(config.config['STFT_WINDOW_LENGTH']),  # 512
        hop_length      = int(config.config['STFT_HOP_LENGTH']),
        window_type     = config.config['STFT_WINDOW_TYPE'],
    )
    nf = stft_params.window_length // 2 + 1  # 257 if window_length=512

    print(f"Using device: {device} | nf={nf} | channels={config.config['MODEL_NUM_CHANNELS']} "f"| hidden={config.config['MODEL_HIDDEN_SIZE']} | stems={num_stems} | loss={loss_fn_name}",flush=True)
        
    # ----------------------------
    # Model
    # ----------------------------
    model = MaskInference.build(
        nf,
        num_audio_channels      = int(config.config['MODEL_NUM_CHANNELS']),
        hidden_size             = int(config.config['MODEL_HIDDEN_SIZE']),
        num_layers              = int(config.config['MODEL_NUM_LAYERS']),
        bidirectional           = bool(config.config['MODEL_BIDIRECTIONAL']),
        dropout                 = float(config.config['MODEL_DROPOUT']),
        num_sources             = num_stems,
        activation              = config.config['MODEL_ACTIVATION'],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.config["LEARNING_RATE"]),
        weight_decay=float(config.config.get("WEIGHT_DECAY", 0.0)),
    )

    use_cuda_amp = (device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda_amp)
    # ----------------------------
    # Effective batch size via GA
    # ----------------------------
    actual_batch_size, accumulation_steps, effective_batch_size = choose_batch_settings(loss_fn_name,num_stems)
    print(
        f"Batch settings: actual_batch  = {actual_batch_size} |"
        f"accum_steps                   = {accumulation_steps} |" 
        f"effective_batch               = {effective_batch_size}",
        flush                           = True
        )
    # ----------------------------
    # Loss function 
    # ----------------------------
    loss_type = str(config.config['MODEL_LOSS_FUNCTION']).lower()
    
    accumulation_steps = math.ceil(effective_batch_size / actual_batch_size)
    # ----------------------------
    # Data
    # ----------------------------
    train_data, val_data = get_data(
        stft_params,
        config.config['MAX_MIXTURES'],
        config.config['COHERENT_PROB']
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        num_workers         = 0,
        batch_size          = actual_batch_size,
        pin_memory          = False,
        prefetch_factor     = None,
        persistent_workers  = False
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_data,
        num_workers         = 0,
        batch_size          = actual_batch_size,
        pin_memory          = False,
        prefetch_factor     = None,
        persistent_workers  = False
    )
    # ----------------------------
    # Gradient accumulation state
    # ----------------------------
    micro_in_accum = 0  # microbatches acumulados desde el último step
    optimizer.zero_grad(set_to_none=True)
    # ----------------------------
    # Train / Val steps
    # ----------------------------
    def train_step(engine, batch):
        nonlocal micro_in_accum
        
        if torch.cuda.is_available():
            print(torch.cuda.memory_allocated() / 1024**3, "GB allocated")
            print(torch.cuda.memory_reserved() / 1024**3, "GB reserved")
        model.train()

        batch = prepare_batch(batch, device=device)
        if engine.state.iteration == 1:

            print("Batch keys:", batch.keys(), flush=True)

            if 'mixture_phase' in batch: print("mixture_phase shape:", batch['mixture_phase'].shape, flush=True)        
            if 'source_phase' in batch: print("source_phase shape:", batch['source_phase'].shape, flush=True)
            if 'source_magnitudes' in batch: print("source_magnitudes shape:", batch['source_magnitudes'].shape, flush=True)
                
        kwargs = build_loss_kwargs(loss_type, batch)

        with torch.amp.autocast('cuda', enabled=use_cuda_amp):

            output = model(batch)
            estimates = output['estimates']
            targets = batch['source_magnitudes']

            if engine.state.iteration == 1:
                em, es = estimates.mean().item(), estimates.std().item()
                tm, ts = targets.mean().item(), targets.std().item()
                print(f"Estimates stats - mean: {em:.4f}, std: {es:.4f}", flush=True)
                print("est shape:", estimates.shape, flush=True)
                print("tgt shape:", targets.shape, flush=True)

        loss = compute_training_loss(loss_type, output, batch, kwargs) / accumulation_steps

        scaler.scale(loss).backward()

        micro_in_accum += 1
        if (micro_in_accum % accumulation_steps) == 0:
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(),float(config.config.get("GRADIENT_CLIP", 1.0)))
                
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            micro_in_accum = 0

        return {'loss': loss.item() * accumulation_steps}

    def val_step(engine, batch):

        model.eval()
        batch = prepare_batch(batch, device=device)

        kwargs = build_loss_kwargs(loss_type, batch)

        with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_cuda_amp):
            output = model(batch)
            estimates = output['estimates']
            targets = batch['source_magnitudes']
            loss = compute_training_loss(loss_type, output, batch, kwargs)

        return {'loss': loss.item()}
    # ----------------------------
    # Ignite engines
    # ----------------------------
    trainer, validator = nussl.ml.train.create_train_and_validation_engines(train_step, val_step, device=device)
    # ----------------------------
    # Output / checkpoints folder
    # ----------------------------
    output_folder = checkpoint_dir(loss_fn_name, num_stems)
    output_folder.mkdir(parents=True, exist_ok=True)

    snapshot = training_config_snapshot(loss_fn_name, num_stems)
    save_json(output_folder / "training_config.json", snapshot)

    nussl.ml.train.add_stdout_handler(trainer, validator)

    Average(output_transform=lambda x: x['loss']).attach(validator, 'val_loss')

    def score_function(engine):
        return -engine.state.metrics['val_loss']


    to_save = {
        'model':            model,
        'optimizer':        optimizer,
        'trainer':          trainer,
        'scaler':           scaler,
    }

    best_model_handler = Checkpoint(
        to_save                     = to_save,
        save_handler                = DiskSaver(str(output_folder), create_dir=True, require_empty=False),
        filename_prefix             = 'best',
        n_saved                     = 1,  # <- SOLO conservamos el mejor checkpoint
        global_step_transform       = global_step_from_engine(trainer),
        score_function              = score_function,
        score_name                  = 'val_loss'
    )

    validator.add_event_handler(Events.COMPLETED, best_model_handler)# Este handler se ejecuta al terminar cada validación.Si la val_loss mejora, sustituye el checkpoint anterior.
    # ============================================================

    early_stopper = EarlyStopping(
        patience                    = config.config['EARLY_STOPPING_PATIENCE'],
        score_function              = score_function,
        trainer                     = trainer
    )
    validator.add_event_handler(Events.COMPLETED, early_stopper)

    @validator.on(Events.COMPLETED)
    def print_val_loss(engine):

        print(f"[Validator] Epoch {trainer.state.epoch} - val_loss: {engine.state.metrics['val_loss']:.6f}", flush=True)
        if best_model_handler.last_checkpoint is not None:
            print(f"[BEST CHECKPOINT] Guardado/actualizado en: {best_model_handler.last_checkpoint}", flush=True)# Mensaje visual para saber dónde ha quedado el mejor modelo.last_checkpoint apunta al fichero más reciente guardado por este handler.
        # ============================================================

    iter_losses, epoch_losses = [], []

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_iter_loss(engine):

        out = engine.state.output
        if isinstance(out, dict) and 'loss' in out: iter_losses.append(out['loss'])
            
    @trainer.on(Events.EPOCH_COMPLETED)
    def run_validation(engine):

        nonlocal micro_in_accum

        if micro_in_accum != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),float(config.config.get("GRADIENT_CLIP", 1.0)),)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            micro_in_accum = 0

        validator.run(val_dataloader)        # Run validation once per epoch
        n = engine.state.epoch_length       # Mean train loss over the real epoch_length (not len(dataloader))
        mean_loss = sum(iter_losses[-n:]) / n if n and len(iter_losses) >= n else float('nan')
        epoch_losses.append(mean_loss)

        val_loss = validator.state.metrics.get('val_loss', float('nan'))
        print(f"[Epoch {engine.state.epoch}] train_loss={mean_loss:.6f} | val_loss={val_loss:.6f}", flush=True)
    # ----------------------------
    # Train
    # ----------------------------
    epoch_length = min(int(config.config['EPOCH_LENGTH']), len(train_dataloader))
    trainer.run(train_dataloader,epoch_length=epoch_length,max_epochs=int(config.config['MAX_EPOCHS']))

    best_checkpoint_path = best_model_handler.last_checkpoint# ANTES de calcular métricas finales, recargamos el MEJOR modelo guardado en validación. Aseguramos de evaluar el mejor y no simplemente el último que quedó en memoria al parar el training.

    if best_checkpoint_path is None: print("[WARNING] No se encontró ningún best checkpoint. Se evaluará el modelo final en memoria.", flush=True)
    else:
        print(f"[LOADING BEST MODEL] Cargando mejor checkpoint desde: {best_checkpoint_path}", flush=True)

        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])# Checkpoint de Ignite guarda cada objeto por clave:checkpoint['model'], checkpoint['optimizer'], etc.

        # Si se quiere continuar el entrenamiento:
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # scaler.load_state_dict(checkpoint['scaler'])

    # ----------------------------
    # Eval metrics + save results
    # ----------------------------
    sisdr_por_muestra = calcular_sisdr_por_muestra(model, val_dataloader)
    metricas_dict = calcular_metricas_globales(model, val_dataloader)

    guardar_resultados(
        model_name          = f'{loss_fn_name}_{num_stems}stems',
        metrics             = metricas_dict,
        sisdr_list          = sisdr_por_muestra,
        loss_history        = {'iter': iter_losses, 'epoch': epoch_losses}
    )