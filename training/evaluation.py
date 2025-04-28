import nussl
from nussl.datasets import transforms as nussl_tfm
from pathlib import Path
import json
import glob
import numpy as np
from utils.config import config




def evaluation(frames,separator,output_folder='JSONs'):
    output_folder = Path(__file__).parent / output_folder
    output_folder.mkdir(parents=True, exist_ok=True)  # crea carpeta JSONS si no existe

    tfm = nussl_tfm.Compose([
        nussl_tfm.SumSources([['bass', 'drums', 'other']]),
    ])

    test_dataset = nussl.datasets.MUSDB18(subsets=['test'], transform=tfm)

    # Evaluamos
    for i in range(frames):

        item = test_dataset[i]
        separator.audio_signal = item['mix']
        estimates = separator()

        source_keys = list(item['sources'].keys())
        estimates = {
            'vocals': estimates[0],
            'bass+drums+other': item['mix'] - estimates[0]
        }

        sources = [item['sources'][k] for k in source_keys]
        estimates = [estimates[k] for k in source_keys]

        evaluator = nussl.evaluation.BSSEvalScale(
            sources, estimates, source_labels=source_keys
        )
        scores = evaluator.evaluate()

        output_file = output_folder / (sources[0].file_name.replace('wav', 'json'))
        with open(output_file, 'w') as f:
            json.dump(scores, f, indent=4)

    # Leer todos los JSONs de la carpeta correcta
    json_files = glob.glob(str(output_folder / '*.json'))
    df = nussl.evaluation.aggregate_score_files(json_files, aggregator=np.nanmedian)

    df.drop('file', axis=1, inplace=True)

    nussl.evaluation.associate_metrics(separator.model, df, test_dataset)
    report_card = nussl.evaluation.report_card(df, report_each_source=True)
    output_folder = '/home/martin/PycharmProjects/TFG/checkpoints'
    separator.model.save(output_folder + '/final_model.pth')

    print(report_card)