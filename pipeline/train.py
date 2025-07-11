# 1. Imports
import torch
from pathlib import Path
import nussl

from models.mask_inference import MaskInference
from data.data_loader import get_data
from config import config
from commons.metrics import get_loss_fn

stft_params = nussl.STFTParams(
    window_length = config.config['STFT_WINDOW_LENGTH'],
    hop_length = config.config['STFT_HOP_LENGTH'],
    window_type = config.config['STFT_WINDOW_TYPE'],
)
nf = stft_params.window_length // 2 + 1
model = MaskInference.build(
    nf,
    num_audio_channels=1,
    hidden_size=config.config['MODEL_HIDDEN_SIZE'],
    num_layers=config.config['MODEL_NUM_LAYERS'],
    bidirectional=config.config['MODEL_BIDIRECTIONAL'],
    dropout=config.config['MODEL_DROPOUT'],
    num_sources=config.config['MODEL_NUM_SOURCES'],
    activation=config.config['MODEL_ACTIVATION']

)
optimizer = torch.optim.Adam(model.parameters(), lr=config.config['LEARNING_RATE'])
DEVICE = config.config['DEVICE'] if torch.cuda.is_available() else 'cpu'

loss_fn = get_loss_fn(config.config['MODEL_LOSS_FUNCTION'])  # solo cambias esta línea

def train_step(engine, batch):


    optimizer.zero_grad()
    output = model(batch)
    if config.config['MODEL_LOSS_FUNCTION'] == 'lmrs':
        loss = loss_fn()
    loss = loss_fn(output['estimates'],batch['source_magnitudes'])
    loss.backward()
    optimizer.step()

    return {'loss': loss.item()}

def val_step(engine, batch):

    with torch.no_grad():
        output = model(batch)
        loss_fn = nussl.ml.train.loss.L1Loss()
        loss = loss_fn(output['estimates'], batch['source_magnitudes'])
    return {'loss': loss.item()}


def training():

    train_data, val_data = get_data(stft_params, config.config['MAX_MIXTURES'], config.config['COHERENT_PROB'])

    trainer, validator = nussl.ml.train.create_train_and_validation_engines(
        train_step, val_step, device=DEVICE)

    train_dataloader = torch.utils.data.DataLoader(
        train_data, num_workers=1, batch_size=config.config['BATCH_SIZE'])

    val_dataloader = torch.utils.data.DataLoader(
        val_data, num_workers=1, batch_size=config.config['BATCH_SIZE'])
    loss_fn = config.config['MODEL_LOSS_FUNCTION']
    output_folder = Path(f'{loss_fn} checkpoints/{config.config["MODEL_NUM_SOURCES"]}stems').absolute()
    output_folder.mkdir(parents=True, exist_ok=True)

    nussl.ml.train.add_stdout_handler(trainer, validator)

    nussl.ml.train.add_validate_and_checkpoint(output_folder, model, optimizer,
                                                train_data, trainer, val_dataloader, validator)

    trainer.run(
        train_dataloader,
        epoch_length = config.config['EPOCH_LENGTH'],
        max_epochs = config.config['MAX_EPOCHS']
    )





