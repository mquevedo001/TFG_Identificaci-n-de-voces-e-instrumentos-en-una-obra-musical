from pipeline.evaluation import evaluation
from pipeline.train import training
from commons.generation import generate_mix_generations
from commons.folder_utils import * 
from data.test_loader import load_test_dataset
from config import config
from scripts.evaluate_models import evaluate_model
from pipeline.deployment import deploy
import argparse
import nussl
from pathlib import Path
import torch
import os

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.empty_cache()
    print(f"Usando dispositivo: {device}")
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'eval', 'deploy'], required=True)


    parser.add_argument('--input', type=str, help='Ruta del archivo de audio para deploy.')
    parser.add_argument('--maxepochs', type=str, help='Numero de iteraciones de entrenamiento')
    parser.add_argument('--num_generations', type=str, help='Numero de mezclas coherentes/incoherentes a generar')
    parser.add_argument('--random_deploy',type=str, help='Argumento para separación de pista aleatoria de MUSDB')
    parser.add_argument('--lossfn',type=str,help='Argumento para selección de la función de pérdida')
    parser.add_argument('--numsources',type=int,help='Número de fuentes a separar (2/4)')
    parser.add_argument('--model_group',type=str,help='Grupo de modelos para evaluación ()')

    loss_fn = config.config['MODEL_LOSS_FUNCTION']

    args = parser.parse_args()

    if args.mode == 'train':

        if not args.maxepochs: raise ValueError('Es necesario el argumento del valor del número de iteraciones para el entrenamiento  [ --maxepochs <value> ]')        
        if not args.lossfn: raise ValueError('Es necesario indicar una función de pérdida')
        if not args.numsources: raise ValueError('Es necesario indicar el número de fuentes a separar')

        config.config['MAX_EPOCHS'] = int(args.maxepochs)
        config.config['MODEL_LOSS_FUNCTION'] = str(args.lossfn)
        config.config['MODEL_NUM_SOURCES'] = int(args.numsources)
        training()


    elif args.mode == 'eval': 

        if not args.lossfn: raise ValueError('Es necesario indicar una función de pérdida')
        if not args.numsources: raise ValueError('Es necesario indicar el número de fuentes a evaluar')
        if args.numsources not in (2,4): raise ValueError("El numero de fuentes debe de ser 2 o 4")

        config.config['MODEL_NUM_SOURCES'] = int(args.numsources)
        config.config['MODEL_LOSS_FUNCTION'] = str(args.lossfn)

        TEST_DATA_PATH =  Path(config.config["TEST_DATA_PATH"])
        
        dataset = load_test_dataset(TEST_DATA_PATH)
        
        evaluate_model(str(args.lossfn),dataset,source_counts=int(args.numsources))
        
        
    elif args.mode == 'deploy':

        if not args.lossfn: raise ValueError('Es necesario indicar una función de pérdida')
        if not args.numsources: raise ValueError('Es necesario indicar el numero de fuentes a separar')

        config.config['MODEL_LOSS_FUNCTION'] = args.lossfn
        config.config['MODEL_NUM_SOURCES'] = args.numsources

        output_folder = Path('.') / 'Results'
        output_folder = Path.absolute(output_folder)

        if not args.input: 
            deploy(output_folder,None)
        else: 
            deploy(output_folder,args.input)


    #elif args.mode == 'mix_generation':
    #    if not args.num_generations: raise ValueError('Añade el numero de mezclas a generar <numero_de_mezclas>')
    #    generate_mix_generations( int(args.num_generations) )

            
if __name__ == '__main__':
    main()




