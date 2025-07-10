import json
import nussl







def evaluarFrames(frames,test_dataset,output_folder,separator):

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
