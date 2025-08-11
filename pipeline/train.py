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

stft_params = nussl.STFTParams(
    window_length = config.config['STFT_WINDOW_LENGTH'],
    hop_length = config.config['STFT_HOP_LENGTH'],
    window_type = config.config['STFT_WINDOW_TYPE'],
)
nf = stft_params.window_length // 2 + 1

model = MaskInference.build(
    nf,
    num_audio_channels=config.config['MODEL_NUM_CHANNELS'],
    hidden_size=config.config['MODEL_HIDDEN_SIZE'],
    num_layers=config.config['MODEL_NUM_LAYERS'],
    bidirectional=config.config['MODEL_BIDIRECTIONAL'],
    dropout=config.config['MODEL_DROPOUT'],
    num_sources=4,
    activation=config.config['MODEL_ACTIVATION']

)
optimizer = torch.optim.Adam(model.parameters(), lr=config.config['LEARNING_RATE'])
DEVICE = config.config['DEVICE'] if torch.cuda.is_available() else 'cpu'
num_stems = config.config['MODEL_NUM_SOURCES']


def train_step(engine, batch):
    model.train()
    optimizer.zero_grad()

    batch = prepare_batch(batch, device=DEVICE)


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
    batch = prepare_batch(batch, device=DEVICE)
    with torch.no_grad():
        output = model(batch)
        loss_fn = nussl.ml.train.loss.L1Loss()
        loss = loss_fn(output['estimates'], batch['source_magnitudes'])
    return {'loss': loss.item()}



def training():
    print(f"[DEBUG] model = {model}")
    dummy_input = {
        'mix_magnitude': torch.rand(1, 10, 257, 1)
    }
    output = model(dummy_input)

    loss_fn = config.config['MODEL_LOSS_FUNCTION']
    train_data, val_data = get_data(
        stft_params,
        config.config['MAX_MIXTURES'],
        config.config['COHERENT_PROB']
    )

    trainer, validator = nussl.ml.train.create_train_and_validation_engines(
        train_step, val_step, device=DEVICE
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_data, num_workers=0, batch_size=config.config['BATCH_SIZE']
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_data, num_workers=0, batch_size=config.config['BATCH_SIZE']
    )

    numSources = config.config['MODEL_NUM_SOURCES']

    output_folder = Path('.') / 'checkpoints' / 'Mis_modelos' / f'{loss_fn} checkpoints' / f'{numSources}stems'
    output_folder.mkdir(parents=True, exist_ok=True)

    nussl.ml.train.add_stdout_handler(trainer, validator)

    # 👇 VALIDACIÓN Y CHECKPOINTING
    nussl.ml.train.add_validate_and_checkpoint(
        output_folder, model, optimizer,
        train_data, trainer, val_dataloader, validator
    )

    # ✅ 1. Añadir métrica de validación para usar con EarlyStopping
    from ignite.metrics import Average
    Average(output_transform=lambda x: x['loss']).attach(validator, 'val_loss')

    # ✅ 2. Definir función de puntuación
    def score_function(engine):
        val_loss = engine.state.metrics['val_loss']
        return -val_loss  # porque queremos minimizar la pérdida

    # ✅ 3. Crear handler de early stopping
    early_stopper = EarlyStopping(
        patience=20,  # Número de épocas sin mejora
        score_function=score_function,
        trainer=trainer
    )

    # ✅ 4. Añadir early stopper al validador
    validator.add_event_handler(Events.COMPLETED, early_stopper)

    # ✅ 5. (Opcional) imprimir cuándo se activa
    @validator.on(Events.COMPLETED)
    def print_val_loss(engine):
        print(f"[Validator] Epoch {trainer.state.epoch} - val_loss: {engine.state.metrics['val_loss']:.6f}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_validation(engine):
        validator.run(val_dataloader)

    trainer.run(
        train_dataloader,
        epoch_length=config.config['EPOCH_LENGTH'],
        max_epochs=config.config['MAX_EPOCHS']
    )

    plt.plot(trainer.state.iter_history['loss'])
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f'{loss_fn} Train Loss')
    output_folder = Path('.') / 'Results' / 'Graphs'
    output_folder.mkdir(parents=True, exist_ok=True)
    save_path = Path(f'./resultados_modelos/individuales/{loss_fn}_{num_stems}stems')
    model_name = f'{loss_fn}_{num_stems}stems'
    lista_de_losses = trainer.state.loss_history['loss']
    sisdr_por_muestra = calcular_sisdr_por_muestra(model, val_dataloader)
    metricas_dict = calcular_metricas_globales(model,val_dataloader)
    lista_de_losses = trainer.state.iter_history['loss']

    guardar_resultados(
        model_name=model_name,
        metrics=metricas_dict,
        sisdr_list=sisdr_por_muestra,
        loss_history=lista_de_losses
    )







