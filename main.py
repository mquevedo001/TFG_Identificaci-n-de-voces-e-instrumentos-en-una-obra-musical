from data.data_loader import  get_data
from models.mask_inference import MaskInference
from training.evaluation import evaluation
from training.train import train_data
from training.train import training
from utils.config import config
from deployment import deploy
import os
import argparse
import nussl
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'eval', 'deploy'], required=True)
    parser.add_argument('--input', type=str, help='Ruta del archivo de audio para deploy.')
    parser.add_argument('--maxepochs', type=str, help='Ruta del archivo de audio para deploy.')

    args = parser.parse_args()

    if args.mode == 'train':
        if not args.maxepochs:
            raise ValueError('Es necesario el argumento del valor del número de iteraciones para el entrenamiento  [ --maxepochs <value> ]')
        else:
            config.config['MAX_EPOCHS'] = int(args.maxepochs)
            training()

    elif args.mode == 'eval':
        output_folder =   Path('.') / 'models' / 'checkpoints'
        output_folder = Path.absolute(output_folder)
        model_path = output_folder / 'best.model.pth'

        if not model_path.exists():
            raise FileNotFoundError(
                f"Modelo no encontrado en {model_path.resolve()}. Entrena el modelo primero usando '--mode train'.")

        separator = nussl.separation.deep.DeepMaskEstimation(
            nussl.AudioSignal(),
            model_path=str(model_path.resolve()),  # <-- importante
            device=config.config['DEVICE'],
        )

        evaluation(frames=5, separator=separator)
    elif args.mode == 'deploy':
        if not args.input:
            raise ValueError('Debes pasar un archivo de audio usando --input <ruta_al_archivo>')

        output_folder = Path('.') / 'Results'
        output_folder = Path.absolute(output_folder)

        deploy(args.input, output_folder)

if __name__ == '__main__':
    main()
