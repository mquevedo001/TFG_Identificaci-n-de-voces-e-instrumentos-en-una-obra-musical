from nussl.datasets import transforms as nussl_tfm
from utils.config import config
from common import data,utils


def get_data(stft_params , max_mixtures = config.config['MAX_MIXTURES'], coherent_prob = config.config['COHERENT_PROB'] ):

    utils.logger()

    tfm = nussl_tfm.Compose([
        nussl_tfm.SumSources([['vocals'], ['bass', 'drums', 'other']]),
        nussl_tfm.MagnitudeSpectrumApproximation(),
        nussl_tfm.ToSeparationModel(),
    ])

    train_folder = "mix_generation/foreground"
    val_folder = "~/.nussl/tutorial/valid"
    print("Training model with training mixes in in:",train_folder + "\n")
    print("Validating data with validation mixes in:",val_folder + "\n")

    train_data = data.on_the_fly(stft_params, transform = tfm,
                                 fg_path = train_folder, num_mixtures = max_mixtures, coherent_prob = coherent_prob)
    val_data = data.on_the_fly(stft_params, transform = tfm,
                               fg_path = val_folder, num_mixtures=10, coherent_prob = coherent_prob)

    return train_data,val_data