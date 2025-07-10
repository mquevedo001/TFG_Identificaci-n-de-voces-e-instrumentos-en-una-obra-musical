import nussl
from nussl.datasets import transforms as nussl_tfm
from pathlib import Path
import glob
import numpy as np
from commons.metrics import evaluarFrames

def evaluation(frames,separator,output_folder='JSONs'):
    output_folder = Path(__file__).parent / output_folder
    output_folder.mkdir(parents=True, exist_ok=True)  # crea carpeta JSONS si no existe

    tfm = nussl_tfm.Compose([
        nussl_tfm.SumSources([['bass', 'drums', 'other']]),
    ])

    test_dataset = nussl.datasets.MUSDB18(subsets=['test'], transform=tfm)

    # Evaluamos
    evaluarFrames(frames,test_dataset,output_folder,separator)

    # Leer todos los JSONs de la carpeta correcta
    json_files = glob.glob(str(output_folder / '*.json'))
    df = nussl.evaluation.aggregate_score_files(json_files, aggregator=np.nanmedian)

    df.drop('file', axis=1, inplace=True)

    nussl.evaluation.associate_metrics(separator.model, df, test_dataset)
    report_card = nussl.evaluation.report_card(df, report_each_source=True)


    print(report_card)