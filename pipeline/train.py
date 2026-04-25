# train.py
import math
from pathlib import Path

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

# ============================================================
# =====================  NUEVO  ==============================
# ============================================================
# IMPORTS para guardar checkpoints del mejor modelo
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
# ============================================================


def training():
    # ----------------------------
    # Config / device
    # ----------------------------
    loss_fn_name = config.config['MODEL_LOSS_FUNCTION']
    num_stems = int(config.config['MODEL_NUM_SOURCES'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # STFT
    stft_params = nussl.STFTParams(
        window_length=int(config.config['STFT_WINDOW_LENGTH']),  # 512
        hop_length=int(config.config['STFT_HOP_LENGTH']),
        window_type=config.config['STFT_WINDOW_TYPE'],
    )
    nf = stft_params.window_length // 2 + 1  # 257 if window_length=512

    print(
        f"Using device: {device} | nf={nf} | channels={config.config['MODEL_NUM_CHANNELS']} "
        f"| hidden={config.config['MODEL_HIDDEN_SIZE']} | stems={num_stems} | loss={loss_fn_name}",
        flush=True
    )

    # ----------------------------
    # Model
    # ----------------------------
    model = MaskInference.build(
        nf,
        num_audio_channels=int(config.config['MODEL_NUM_CHANNELS']),
        hidden_size=int(config.config['MODEL_HIDDEN_SIZE']),
        num_layers=int(config.config['MODEL_NUM_LAYERS']),
        bidirectional=bool(config.config['MODEL_BIDIRECTIONAL']),
        dropout=float(config.config['MODEL_DROPOUT']),
        num_sources=num_stems,
        activation=config.config['MODEL_ACTIVATION'],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.config['LEARNING_RATE']))

    # AMP
    use_cuda_amp = (device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda_amp)

    # ----------------------------
    # Effective batch size via GA
    # ----------------------------
    effective_batch_size = 100

    # Para 3060Ti + STFT512:
    # - 2 stems: suele aguantar 32
    # - 4 stems: a veces 16-24 es más seguro
    actual_batch_size = int(config.config.get('BATCH_SIZE', 32))
    if num_stems == 4:
        actual_batch_size = min(actual_batch_size, 24)

    accumulation_steps = math.ceil(effective_batch_size / actual_batch_size)
    effective_batch_real = accumulation_steps * actual_batch_size

    print(
        f"GA: actual_batch={actual_batch_size} | accum_steps={accumulation_steps} "
        f"| effective_batch_real={effective_batch_real}",
        flush=True
    )

    # ----------------------------
    # Loss function 
    # ----------------------------
    loss_type = str(config.config['MODEL_LOSS_FUNCTION']).lower()
    #[TEST]
    effective_batch_size = 100

    if loss_type == 'deep_feature_emd' or loss_type == 'deep_feature':
        actual_batch_size = 4
    else:
        actual_batch_size = 32

    accumulation_steps = math.ceil(effective_batch_size / actual_batch_size)

    # ----------------------------
    # Data
    # ----------------------------
    train_data, val_data = get_data(
        stft_params,
        config.config['MAX_MIXTURES'],
        config.config['COHERENT_PROB']
    )

    # Windows stability: pin_memory False + low prefetch
    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        num_workers=0,
        batch_size=actual_batch_size,
        pin_memory=False,
        prefetch_factor=None,
        persistent_workers=False
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_data,
        num_workers=0,
        batch_size=actual_batch_size,
        pin_memory=False,
        prefetch_factor=None,
        persistent_workers=False
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

        # batch to device
        print(batch.keys())
        batch = prepare_batch(batch, device=device)
        if engine.state.iteration == 1:
            print("Batch keys:", batch.keys(), flush=True)
            if 'mixture_phase' in batch:
                print("mixture_phase shape:", batch['mixture_phase'].shape, flush=True)
            if 'source_phase' in batch:
                print("source_phase shape:", batch['source_phase'].shape, flush=True)
            if 'source_magnitudes' in batch:
                print("source_magnitudes shape:", batch['source_magnitudes'].shape, flush=True)
        # kwargs depende del loss
        kwargs = {}
        if loss_type == 'lpsa_phase':
            if 'mixture_phase' not in batch:
                raise ValueError("El batch no contiene 'mixture_phase'.")
            if 'source_phase' not in batch:
                raise ValueError("El batch no contiene 'source_phase'.")
            kwargs['mixture_phase'] = batch['mixture_phase']
            kwargs['source_phase'] = batch['source_phase']
        if loss_type in ['l_mrs', 'lmrs']:
            if 'mixture_magnitude' not in batch:
                raise ValueError("El batch no contiene 'mixture_magnitude'.")
            kwargs['mix_mag'] = batch['mixture_magnitude']

        if loss_type == 'l_mrs':
            kwargs['mixture_phase'] = batch['mixture_phase']

        # forward + loss
        with torch.amp.autocast('cuda', enabled=use_cuda_amp):
            output = model(batch)
            estimates = output['estimates']
            targets = batch['source_magnitudes']

            # sanity check 1 vez por run (iteration global = 1)
            if engine.state.iteration == 1:
                em, es = estimates.mean().item(), estimates.std().item()
                tm, ts = targets.mean().item(), targets.std().item()
                #print(f"[DEBUG] estimates mean/std: {em:.6g} {es:.6g}", flush=True)
                #print(f"[DEBUG] targets   mean/std: {tm:.6g} {ts:.6g}", flush=True)
                #print(f"[DEBUG] estimates shape: {tuple(estimates.shape)} | targets shape: {tuple(targets.shape)}", flush=True)
                print("est shape:", estimates.shape, flush=True)
                print("tgt shape:", targets.shape, flush=True)
            loss_fn = get_loss_fn(loss_type, kwargs)

            # ---- Correct GA scaling:
            loss = loss_fn(estimates, targets) / accumulation_steps

        # backward
        scaler.scale(loss).backward()

        micro_in_accum += 1
        if (micro_in_accum % accumulation_steps) == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            micro_in_accum = 0

        # report *unscaled* loss for readability

        return {'loss': loss.item() * accumulation_steps}

    def val_step(engine, batch):
        model.eval()
        batch = prepare_batch(batch, device=device)

        kwargs = {}
        if loss_type == 'lpsa_phase':
            if 'mixture_phase' not in batch:
                raise ValueError("El batch no contiene 'mixture_phase'.")
            if 'source_phase' not in batch:
                raise ValueError("El batch no contiene 'source_phase'.")
            kwargs['mixture_phase'] = batch['mixture_phase']
            kwargs['source_phase'] = batch['source_phase']
        if loss_type in ['l_mrs', 'lmrs']:
            if 'mixture_magnitude' not in batch:
                raise ValueError("El batch no contiene 'mixture_magnitude'.")
            kwargs['mix_mag'] = batch['mixture_magnitude']

        with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_cuda_amp):
            output = model(batch)
            estimates = output['estimates']
            targets = batch['source_magnitudes']
            loss_fn = get_loss_fn(loss_type, kwargs)
            loss = loss_fn(estimates, targets)

        return {'loss': loss.item()}

    # ----------------------------
    # Ignite engines
    # ----------------------------
    trainer, validator = nussl.ml.train.create_train_and_validation_engines(
        train_step, val_step, device=device
    )

    # ----------------------------
    # Output / checkpoints folder
    # ----------------------------
    output_folder = Path('.') / 'checkpoints' / 'Mis_modelos' / f'{loss_fn_name} checkpoints' / f'{num_stems}stems'
    output_folder.mkdir(parents=True, exist_ok=True)

    nussl.ml.train.add_stdout_handler(trainer, validator)

    # Validation metric for early stopping
    Average(output_transform=lambda x: x['loss']).attach(validator, 'val_loss')

    def score_function(engine):
        return -engine.state.metrics['val_loss']

    # ============================================================
    # =====================  NUEVO  ==============================
    # ============================================================
    # Guardaremos SIEMPRE el mejor checkpoint según val_loss.
    # Como score_function devuelve -val_loss, "más alto" = "mejor".
    to_save = {
        'model': model,
        'optimizer': optimizer,
        'trainer': trainer,
        'scaler': scaler,
    }

    best_model_handler = Checkpoint(
        to_save=to_save,
        save_handler=DiskSaver(str(output_folder), create_dir=True, require_empty=False),
        filename_prefix='best',
        n_saved=1,  # <- SOLO conservamos el mejor checkpoint
        global_step_transform=global_step_from_engine(trainer),
        score_function=score_function,
        score_name='val_loss'
    )

    # IMPORTANTE:
    # Este handler se ejecuta al terminar cada validación.
    # Si la val_loss mejora, sustituye el checkpoint anterior.
    validator.add_event_handler(Events.COMPLETED, best_model_handler)
    # ============================================================

    early_stopper = EarlyStopping(
        patience=10,
        score_function=score_function,
        trainer=trainer
    )
    validator.add_event_handler(Events.COMPLETED, early_stopper)

    @validator.on(Events.COMPLETED)
    def print_val_loss(engine):
        # Important: flush for subprocess tee
        print(f"[Validator] Epoch {trainer.state.epoch} - val_loss: {engine.state.metrics['val_loss']:.6f}", flush=True)

        # ============================================================
        # =====================  NUEVO  ==============================
        # ============================================================
        # Mensaje visual para saber dónde ha quedado el mejor modelo.
        # last_checkpoint apunta al fichero más reciente guardado por este handler.
        if best_model_handler.last_checkpoint is not None:
            print(f"[BEST CHECKPOINT] Guardado/actualizado en: {best_model_handler.last_checkpoint}", flush=True)
        # ============================================================

    iter_losses, epoch_losses = [], []

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_iter_loss(engine):
        out = engine.state.output
        if isinstance(out, dict) and 'loss' in out:
            iter_losses.append(out['loss'])

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_validation(engine):
        nonlocal micro_in_accum

        # Flush pending grads if epoch ended mid-window
        if micro_in_accum != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            micro_in_accum = 0

        # Run validation once per epoch
        validator.run(val_dataloader)

        # Mean train loss over the real epoch_length (not len(dataloader))
        n = engine.state.epoch_length
        mean_loss = sum(iter_losses[-n:]) / n if n and len(iter_losses) >= n else float('nan')
        epoch_losses.append(mean_loss)

        val_loss = validator.state.metrics.get('val_loss', float('nan'))
        print(f"[Epoch {engine.state.epoch}] train_loss={mean_loss:.6f} | val_loss={val_loss:.6f}", flush=True)

    # ----------------------------
    # Train
    # ----------------------------
    epoch_length = min(int(config.config['EPOCH_LENGTH']), len(train_dataloader))
    trainer.run(
        train_dataloader,
        epoch_length=epoch_length,
        max_epochs=int(config.config['MAX_EPOCHS'])
    )

    # ============================================================
    # =====================  NUEVO / TOCADO  =====================
    # ============================================================
    # ANTES de calcular métricas finales, recargamos el MEJOR modelo
    # guardado en validación. Aseguramos de evaluar el mejor y no
    # simplemente el último que quedó en memoria al parar el training.
    best_checkpoint_path = best_model_handler.last_checkpoint

    if best_checkpoint_path is None:
        print("[WARNING] No se encontró ningún best checkpoint. Se evaluará el modelo final en memoria.", flush=True)
    else:
        print(f"[LOADING BEST MODEL] Cargando mejor checkpoint desde: {best_checkpoint_path}", flush=True)
        checkpoint = torch.load(best_checkpoint_path, map_location=device)

        # Importante:
        # Checkpoint de Ignite guarda cada objeto por clave:
        # checkpoint['model'], checkpoint['optimizer'], etc.
        model.load_state_dict(checkpoint['model'])

        # Si se quiere continuar el entrenamiento:
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # scaler.load_state_dict(checkpoint['scaler'])
        #
        # Para evaluación final, con cargar model suele ser suficiente.
    # ============================================================

    # ----------------------------
    # Eval metrics + save results
    # ----------------------------
    sisdr_por_muestra = calcular_sisdr_por_muestra(model, val_dataloader)
    metricas_dict = calcular_metricas_globales(model, val_dataloader)

    guardar_resultados(
        model_name=f'{loss_fn_name}_{num_stems}stems',
        metrics=metricas_dict,
        sisdr_list=sisdr_por_muestra,
        loss_history={'iter': iter_losses, 'epoch': epoch_losses}
    )