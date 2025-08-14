# 1. Imports
import torch
from pathlib import Path
import nussl
import matplotlib.pyplot as plt

from models.Mi_modelo.mask_inference import MaskInference
from data.data_loader import get_data
from config import config
from commons.metrics import get_loss_fn
from commons.plotting import guardar_resultados, calcular_metricas_globales
from commons.plotting import calcular_sisdr_por_muestra
from commons.model_utils import prepare_batch

from ignite.handlers import EarlyStopping
from ignite.engine import Events

def training():

    loss_fn_name = config.config['MODEL_LOSS_FUNCTION']
    num_stems = config.config['MODEL_NUM_SOURCES']
    device = config.config['DEVICE'] if torch.cuda.is_available() else 'cpu'

    stft_params = nussl.STFTParams(
        window_length=config.config['STFT_WINDOW_LENGTH'],
        hop_length=config.config['STFT_HOP_LENGTH'],
        window_type=config.config['STFT_WINDOW_TYPE'],
    )
    nf = stft_params.window_length // 2 + 1
    print(
        f"nf={nf}, num_audio_channels={config.config['MODEL_NUM_CHANNELS']}, hidden_size={config.config['MODEL_HIDDEN_SIZE']}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MaskInference.build(
        nf,
        num_audio_channels=config.config['MODEL_NUM_CHANNELS'],
        hidden_size=config.config['MODEL_HIDDEN_SIZE'],
        num_layers=config.config['MODEL_NUM_LAYERS'],
        bidirectional=config.config['MODEL_BIDIRECTIONAL'],
        dropout=config.config['MODEL_DROPOUT'],
        num_sources=num_stems,
        activation=config.config['MODEL_ACTIVATION']
    )

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.config['LEARNING_RATE'])

    # === listas para pérdidas ===
    iter_losses = []
    epoch_losses = []

    def train_step(engine, batch):
        model.train()
        optimizer.zero_grad()

        batch = prepare_batch(batch, device=device)
        output = model(batch)

        estimates = output['estimates']
        targets = batch['source_magnitudes']

        loss_type = config.config['MODEL_LOSS_FUNCTION'].lower()
        kwargs = {}

        if loss_type == 'lpsa_phase':
            if 'mixture_phase' not in batch:
                raise ValueError("El batch no contiene 'mixture_phase'.")
            kwargs['mixture_phase'] = batch.get('mixture_phase')

        if loss_type in ['l_mrs', 'lmrs']:
            if 'mixture_magnitude' not in batch:
                raise ValueError("El batch no contiene 'mixture_magnitude'.")
            kwargs['mix_mag'] = batch['mixture_magnitude']

        loss_fn = get_loss_fn(loss_type, kwargs)
        loss = loss_fn(estimates, targets)

        loss.backward()
        optimizer.step()

        return {'loss': loss.item()}

    def val_step(engine, batch):
        model.eval()
        batch = prepare_batch(batch, device=device)
        with torch.no_grad():
            output = model(batch)
            loss_fn = nussl.ml.train.loss.L1Loss()
            loss = loss_fn(output['estimates'], batch['source_magnitudes'])
        return {'loss': loss.item()}

    train_data, val_data = get_data(
        stft_params,
        config.config['MAX_MIXTURES'],
        config.config['COHERENT_PROB']
    )

    trainer, validator = nussl.ml.train.create_train_and_validation_engines(
        train_step, val_step, device=device
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_data, num_workers=0, batch_size=config.config['BATCH_SIZE']
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_data, num_workers=0, batch_size=config.config['BATCH_SIZE']
    )

    output_folder = Path('.') / 'checkpoints' / 'Mis_modelos' / f'{loss_fn_name} checkpoints' / f'{num_stems}stems'
    output_folder.mkdir(parents=True, exist_ok=True)

    nussl.ml.train.add_stdout_handler(trainer, validator)
    nussl.ml.train.add_validate_and_checkpoint(
        output_folder, model, optimizer,
        train_data, trainer, val_dataloader, validator
    )

    from ignite.metrics import Average
    Average(output_transform=lambda x: x['loss']).attach(validator, 'val_loss')

    def score_function(engine):
        return -engine.state.metrics['val_loss']

    early_stopper = EarlyStopping(
        patience=20,
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
        # Guardar pérdida promedio por epoch
        mean_loss = sum(iter_losses[-len(train_dataloader):]) / len(train_dataloader)
        epoch_losses.append(mean_loss)

    @trainer.on(Events.ITERATION_COMPLETED)
    def guardar_loss(engine):
        output = engine.state.output
        if isinstance(output, dict) and 'loss' in output:
            iter_losses.append(output['loss'])

    trainer.run(
        train_dataloader,
        epoch_length=config.config['EPOCH_LENGTH'],
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









