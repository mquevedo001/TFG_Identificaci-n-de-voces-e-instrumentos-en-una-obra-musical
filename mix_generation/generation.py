from pathlib import Path
import nussl
import scaper
import warnings
import numpy as np
import os
import glob

template_event_parameters = {
    'label': ('const', 'vocals'),
    'source_file': ('choose', []),
    'source_time': ('uniform', 0, 7),
    'event_time': ('const', 0),
    'event_duration': ('const', 5.0),
    'snr': ('uniform', -5, 5),
    'pitch_shift': ('uniform', -2, 2),
    'time_stretch': ('uniform', 0.8, 1.2)
}


def incoherent(fg_folder, bg_folder, event_template, seed):
    """
    This function takes the paths to the MUSDB18 source materials, an event template,
    and a random seed, and returns an INCOHERENT mixture (audio + annotations).

    Stems in INCOHERENT mixtures may come from different songs and are not temporally
    aligned.

    Parameters
    ----------
    fg_folder : str
        Path to the foreground source material for MUSDB18
    bg_folder : str
        Path to the background material for MUSDB18 (empty folder)
    event_template: dict
        Dictionary containing a template of probabilistic event parameters
    seed : int or np.random.RandomState()
        Seed for setting the Scaper object's random state. Different seeds will
        generate different mixtures for the same source material and event template.

    Returns
    -------
    mixture_audio : np.ndarray
        Audio signal for the mixture
    mixture_jams : np.ndarray
        JAMS annotation for the mixture
    annotation_list : list
        Simple annotation in list format
    stem_audio_list : list
        List containing the audio signals of the stems that comprise the mixture
    """

    # Create scaper object and seed random state
    sc = scaper.Scaper(
        duration=5.0,
        fg_path=str(fg_folder),
        bg_path=str(bg_folder),
        random_state=seed
    )

    # Set sample rate, reference dB, and channels (mono)
    sc.sr = 44100
    sc.ref_db = -20
    sc.n_channels = 1

    # Copy the template so we can change it
    event_parameters = event_template.copy()

    # Iterate over stem types and add INCOHERENT events
    labels = ['vocals', 'drums', 'bass', 'other']
    for label in labels:
        event_parameters['label'] = ('const', label)
        sc.add_event(**event_parameters)

    # Return the generated mixture audio + annotations
    # while ensuring we prevent audio clipping
    return sc.generate(fix_clipping=True)


def coherent(fg_folder, bg_folder, event_template, seed):
    """
    This function takes the paths to the MUSDB18 source materials and a random seed,
    and returns an COHERENT mixture (audio + annotations).

    Stems in COHERENT mixtures come from the same song and are temporally aligned.

    Parameters
    ----------
    fg_folder : str
        Path to the foreground source material for MUSDB18
    bg_folder : str
        Path to the background material for MUSDB18 (empty folder)
    event_template: dict
        Dictionary containing a template of probabilistic event parameters
    seed : int or np.random.RandomState()
        Seed for setting the Scaper object's random state. Different seeds will
        generate different mixtures for the same source material and event template.

    Returns
    -------
    mixture_audio : np.ndarray
        Audio signal for the mixture
    mixture_jams : np.ndarray
        JAMS annotation for the mixture
    annotation_list : list
        Simple annotation in list format
    stem_audio_list : list
        List containing the audio signals of the stems that comprise the mixture
    """

    # Create scaper object and seed random state
    sc = scaper.Scaper(
        duration=5.0,
        fg_path=str(fg_folder),
        bg_path=str(bg_folder),
        random_state=seed
    )

    # Set sample rate, reference dB, and channels (mono)
    sc.sr = 44100
    sc.ref_db = -20
    sc.n_channels = 1

    # Copy the template so we can change it
    event_parameters = event_template.copy()

    # Instatiate the template once to randomly choose a song,
    # a start time for the sources, a pitch shift and a time
    # stretch. These values must remain COHERENT across all stems
    sc.add_event(**event_parameters)
    event = sc._instantiate_event(sc.fg_spec[0])

    # Reset the Scaper object's the event specification
    sc.reset_fg_event_spec()

    # Replace the distributions for source time, pitch shift and
    # time stretch with the constant values we just sampled, to
    # ensure our added events (stems) are coherent.
    event_parameters['source_time'] = ('const', event.source_time)
    event_parameters['pitch_shift'] = ('const', event.pitch_shift)
    event_parameters['time_stretch'] = ('const', event.time_stretch)

    # Iterate over the four stems (vocals, drums, bass, other) and
    # add COHERENT events.
    labels = ['vocals', 'drums', 'bass', 'other']
    for label in labels:

        # Set the label to the stem we are adding
        event_parameters['label'] = ('const', label)

        # To ensure coherent source files (all from the same song), we leverage
        # the fact that all the stems from the same song have the same filename.
        # All we have to do is replace the stem file's parent folder name from "vocals"
        # to the label we are adding in this iteration of the loop, which will give the
        # correct path to the stem source file for this current label.
        coherent_source_file = event.source_file.replace('vocals', label)
        event_parameters['source_file'] = ('const', coherent_source_file)
        # Add the event using the modified, COHERENT, event parameters
        sc.add_event(**event_parameters)

    # Generate and return the mixture audio, stem audio, and annotations
    return sc.generate(fix_clipping=True)


def generate_mixture(dataset, fg_folder, bg_folder, event_template, seed):
    # hide warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')

        # flip a coint to choose coherent or incoherent mixing
        random_state = np.random.RandomState(seed)

        # generate mixture
        if random_state.rand() > .5:
            data = coherent(fg_folder, bg_folder, event_template, seed)
        else:
            data = incoherent(fg_folder, bg_folder, event_template, seed)

    # unpack the data
    mixture_audio, mixture_jam, annotation_list, stem_audio_list = data

    # convert mixture to nussl format
    mix = dataset._load_audio_from_array(
        audio_data=mixture_audio, sample_rate=dataset.sample_rate
    )

    # convert stems to nussl format
    sources = {}
    ann = mixture_jam.annotations.search(namespace='scaper')[0]
    for obs, stem_audio in zip(ann.data, stem_audio_list):
        key = obs.value['label']
        sources[key] = dataset._load_audio_from_array(
            audio_data=stem_audio, sample_rate=dataset.sample_rate
        )

    # store the mixture, stems and JAMS annotation in the format expected by nussl
    output = {
        'mix': mix,
        'sources': sources,
        'metadata': mixture_jam
    }
    return output


class MixClosure:

    def __init__(self, fg_folder, bg_folder, event_template):
        self.fg_folder = fg_folder
        self.bg_folder = bg_folder
        self.event_template = event_template

    def __call__(self, dataset, seed):
        return generate_mixture(dataset, self.fg_folder, self.bg_folder, self.event_template, seed)



from pathlib import Path

def create_database():
    bg_folder = Path('mix_generation/background')
    fg_folder = Path('mix_generation/foreground')

    musdb_train = nussl.datasets.MUSDB18(download=True, subsets=['train'])

    for item in musdb_train:
        song_name = item['mix'].file_name
        for key, val in item['sources'].items():
            src_path = fg_folder / key
            src_path.mkdir(parents=True, exist_ok=True)
            output_path = src_path / (song_name + '.wav')
            val.write_audio_to_file(str(output_path))

def get_source_files(fg_folder, label='vocals'):
    path = os.path.join(fg_folder, label, '*.wav')
    files = glob.glob(path)
    if not files:
        raise FileNotFoundError(f"No se encontraron archivos .wav en {path}")
    return files

def generate_mix_generations(num_mixtures):
    bg_folder = 'mix_generation/background'
    fg_folder = 'mix_generation/foreground'
    dir = os.listdir(fg_folder)

    if not(os.listdir(bg_folder + '/bass')):
        print("Base de datos vacía, creando...")
        create_database()

    # Obtener lista de archivos .wav de vocals
    source_files = get_source_files(fg_folder, label='vocals')

    # Actualizar el template con la lista real de archivos
    event_template = template_event_parameters.copy()
    event_template['source_file'] = ('choose', source_files)

    # Inicializar la función de mezcla
    mix_func = MixClosure(fg_folder, bg_folder, event_template)

    # Crear las mezclas con NUSSL
    on_the_fly = nussl.datasets.OnTheFly(
        num_mixtures=num_mixtures,
        mix_closure=mix_func
    )

    print(f"¡Generadas {num_mixtures} mezclas con éxito!")

