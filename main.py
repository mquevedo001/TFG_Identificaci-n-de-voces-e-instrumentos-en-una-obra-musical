
from pipeline.evaluation import evaluation
from pipeline.train import training
from commons.generation import generate_mix_generations
from config import config
from pipeline.deployment import deploy
import argparse
import nussl
from pathlib import Path



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'eval', 'deploy','mix_generation'], required=True)


    parser.add_argument('--input', type=str, help='Ruta del archivo de audio para deploy.')
    parser.add_argument('--maxepochs', type=str, help='Numero de iteraciones de entrenamiento')
    parser.add_argument('--num_generations', type=str, help='Numero de mezclas coherentes/incoherentes a generar')
    parser.add_argument('--random_deploy',type=str, help='Argumento para separación de pista aleatoria de MUSDB')

    loss_fn = config.config['MODEL_LOSS_FUNCTION']

    args = parser.parse_args()

    if args.mode == 'train':
        if not args.maxepochs:
            raise ValueError('Es necesario el argumento del valor del número de iteraciones para el entrenamiento  [ --maxepochs <value> ]')
        else:
            config.config['MAX_EPOCHS'] = int(args.maxepochs)
            training()

    elif args.mode == 'eval':
        num_sources = config.config['MODEL_NUM_SOURCES']
        output_folder =   Path('.') /'checkpoints'/f'{num_sources}stems'
        output_folder = Path.absolute(output_folder)
        model_path = output_folder / f'{loss_fn} checkpoints'/'best.model.pth'

        if not model_path.exists():
            raise FileNotFoundError(
                f"Modelo no encontrado en {model_path.resolve()}. Entrena el modelo primero usando '--mode train'.")

        separator = nussl.separation.deep.DeepMaskEstimation(
            nussl.AudioSignal(),
            model_path=str(model_path.resolve()),  
            device=config.config['DEVICE'],
        )

        evaluation(frames=5, separator=separator)
    elif args.mode == 'deploy':

        if not args.input:

            output_folder = Path('.') / 'Results'
            output_folder = Path.absolute(output_folder)
            deploy(output_folder,None)
        if args.input:

            output_folder = Path('.') / 'Results'
            output_folder = Path.absolute(output_folder)
            deploy(output_folder, args.input)

    elif args.mode == 'mix_generation':
        if not args.num_generations:
            raise ValueError('Añade el numero de mezclas a generar <numero_de_mezclas>')
        generate_mix_generations( int(args.num_generations) )

if __name__ == '__main__':
    main()
