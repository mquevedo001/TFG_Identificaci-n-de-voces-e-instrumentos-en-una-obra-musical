import torch
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
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

def training():
    loss_fn_name = config.config['MODEL_LOSS_FUNCTION']
    num_stems = config.config['MODEL_NUM_SOURCES']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    stft_params = nussl.STFTParams(
        window_length=config.config['STFT_WINDOW_LENGTH'],
        hop_length=config.config['STFT_HOP_LENGTH'],
        window_type=config.config['STFT_WINDOW_TYPE'],
    )
    nf = stft_params.window_length // 2 + 1
    print(f"nf={nf}, num_audio_channels={config.config['MODEL_NUM_CHANNELS']}, hidden_size={config.config['MODEL_HIDDEN_SIZE']}")

    model = MaskInference.build(
        nf,
        num_audio_channels=config.config['MODEL_NUM_CHANNELS'],
        hidden_size=config.config['MODEL_HIDDEN_SIZE'],
        num_layers=config.config['MODEL_NUM_LAYERS'],
        bidirectional=config.config['MODEL_BIDIRECTIONAL'],
        dropout=config.config['MODEL_DROPOUT'],
        num_sources=num_stems,
        activation=config.config['MODEL_ACTIVATION']
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.config['LEARNING_RATE'])
    scaler = GradScaler()

    iter_losses, epoch_losses = [], []

    # === Ajustes de gradient accumulation ===
    effective_batch_size = 100
    actual_batch_size = 16
    accumulation_steps = effective_batch_size // actual_batch_size
    print(f"Usando gradient accumulation: {accumulation_steps} pasos para batch efectivo {effective_batch_size}")

    def train_step(engine, batch):
        model.train()
        batch = prepare_batch(batch, device=device)

        with autocast():
            output = model(batch)
            estimates = output['estimates']
            targets = batch['source_magnitudes']

            loss_type = config.config['MODEL_LOSS_FUNCTION'].lower()
            kwargs = {}
            if loss_type == 'lpsa_phase':
                kwargs['mixture_phase'] = batch.get('mixture_phase')
            if loss_type in ['l_mrs', 'lmrs']:
                kwargs['mix_mag'] = batch['mixture_magnitude']

            loss_fn = get_loss_fn(loss_type, kwargs)
            loss = loss_fn(estimates, targets) / accumulation_steps

        scaler.scale(loss).backward()

        if (engine.state.iteration % accumulation_steps) == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        return {'loss': loss.item() * accumulation_steps}

    def val_step(engine, batch):
        model.eval()
        batch = prepare_batch(batch, device=device)
        with torch.no_grad(), autocast():
            output = model(batch)
            loss_fn = nussl.ml.train.loss.L1Loss()
            loss = loss_fn(output['estimates'], batch['source_magnitudes'])
        return {'loss': loss.item()}

    train_data, val_data = get_data(
        stft_params,
        config.config['MAX_MIXTURES'],
        config.config['COHERENT_PROB']
    )

    # 🔹 num_workers=0 para evitar deadlocks
    train_dataloader = torch.utils.data.DataLoader(
        train_data, num_workers=1, batch_size=actual_batch_size, pin_memory=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_data, num_workers=1, batch_size=actual_batch_size, pin_memory=True
    )

    trainer, validator = nussl.ml.train.create_train_and_validation_engines(
        train_step, val_step, device=device
    )

    output_folder = Path('.') / 'checkpoints' / 'Mis_modelos' / f'{loss_fn_name} checkpoints' / f'{num_stems}stems'
    output_folder.mkdir(parents=True, exist_ok=True)

    nussl.ml.train.add_stdout_handler(trainer, validator)
    nussl.ml.train.add_validate_and_checkpoint(
        output_folder, model, optimizer,
        train_data, trainer, val_dataloader, validator
    )

    Average(output_transform=lambda x: x['loss']).attach(validator, 'val_loss')

    def score_function(engine):
        return -engine.state.metrics['val_loss']

    early_stopper = EarlyStopping(
        patience=10,
        score_function=score_function,
        trainer=trainer
    )
    validator.add_event_handler(Events.COMPLETED, early_stopper)

    @validator.on(Events.COMPLETED)
    def print_val_loss(engine):
        print(f"[Validator] Epoch {trainer.state.epoch} - val_loss: {engine.state.metrics['val_loss']:.6f}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_validation(engine):
        validator.run(val_dataloader)
        mean_loss = sum(iter_losses[-len(train_dataloader):]) / len(train_dataloader)
        epoch_losses.append(mean_loss)

        val_loss = validator.state.metrics['val_loss']
        print(f"[Epoch {engine.state.epoch}] train_loss={mean_loss:.6f} | val_loss={val_loss:.6f}")

        # aplicar gradientes pendientes
        if any(p.grad is not None for p in model.parameters()):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        torch.cuda.empty_cache()

    @trainer.on(Events.ITERATION_COMPLETED)
    def guardar_loss(engine):
        output = engine.state.output
        if isinstance(output, dict) and 'loss' in output:
            iter_losses.append(output['loss'])

    trainer.run(
        train_dataloader,
        epoch_length=min(config.config['EPOCH_LENGTH'], len(train_dataloader)),
        max_epochs=config.config['MAX_EPOCHS']
    )

    sisdr_por_muestra = calcular_sisdr_por_muestra(model, val_dataloader)
    metricas_dict = calcular_metricas_globales(model, val_dataloader)

    guardar_resultados(
        model_name=f'{loss_fn_name}_{num_stems}stems',
        metrics=metricas_dict,
        sisdr_list=sisdr_por_muestra,
        loss_history={'iter': iter_losses, 'epoch': epoch_losses}
    )

