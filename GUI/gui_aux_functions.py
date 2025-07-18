import streamlit as st
from config import config

def stft_param_selection():
    st.subheader('Parámetros de la formación de transformadas de Fourier')
    st.selectbox("Tamaño de la ventana STFT", options=[512, 1024, 2048, 4096],
                 index=[512, 1024, 2048, 4096].index(config.config.get('STFT_WINDOW_LENGTH', 1024)),
                 key='stft_window_input')

    st.selectbox("Desplazamiento entre ventanas STFT",
                 options=[config.config.get('STFT_WINDOW_LENGTH', 1024) // 2,
                          config.config.get('STFT_WINDOW_LENGTH', 1024) // 4],
                 key='stft_hop_length_input')

    st.selectbox("Tipo de ventana utilizada para la STFT",
                 options=['hann', 'hamming', 'blackman', 'bartlett', 'kaiser', 'rectangular'],
                 index=0,
                 key='stft_window_type_input')


def train_param_selection():
    st.subheader("Parámetros de entrenamiento")
    st.number_input("Learning Rate", value=float(config.config.get('LEARNING_RATE', 0.001)),
                    format="%.5f", key='learning_rate_input')
    st.number_input("Batch Size", min_value=1, value=int(config.config.get('BATCH_SIZE', 16)),
                    key='batch_size_input')
    st.number_input("Épocas", min_value=1, value=int(config.config.get('MAX_EPOCHS', 50)),
                    key='epochs_input')
    st.number_input("Tamaño de las épocas", min_value=10, value=int(config.config['EPOCH_LENGTH']),
                    key='epochs_length_input')
    st.number_input("Gradient clip", min_value=0.0, value=float(config.config['GRADIENT_CLIP']),
                    format="%.2f", key='gradient_clip_input')
    st.number_input("Learning Rate Schedule", min_value=10,
                    value=int(config.config['LEARNING_RATE_SCHEDULE']),
                    key='learning_rate_schedule_input')
    st.number_input("Early Stopping Patience", min_value=0,
                    value=int(config.config['EARLY_STOPPING_PATIENCE']),
                    key='early_stopping_patience_input')
    st.number_input("Weight decay", min_value=0.0, value=float(config.config['WEIGHT_DECAY']),
                    format="%.6f", key='weight_decay_input')


def mix_gen_param_selection():
    st.subheader("Parámetros de generación de mezclas")
    st.number_input("Coherent prob", min_value=0.0, max_value=1.0,
                    value=float(config.config['COHERENT_PROB']), key='coherent_prob_input')


def model_param_selection():
    st.subheader('Parámetros intrínsecos modelo')
    st.selectbox("Número de canáles de entrada (Mono/Stereo)",
                 options=['1,2'], key='num_channels_input')
    st.number_input("Porcentaje de dropout", min_value=0, max_value=100,
                    value=int(config.config['MODEL_DROPOUT']), key='dropout_input')
    st.selectbox("Funcion de activación para la capa de salida",
                 options=['softmax', 'relu', 'sigmoid', 'leaky_relu'],
                 key='activation_func_input')
    st.selectbox("Numero de frames para la evaluación del modelo",
                 options=[0, 5, 50, 100, 500, 1000],
                 index=[0, 5, 50, 100, 500, 1000].index(config.config.get('EVALUATOR_FRAMES', 100)),
                 key='evaluator_frames_input')


def fx_param_selection():
    st.subheader('Parámetros de efectos para entrenamiento y de audio')

    st.number_input("Prob. de aplicar efectos (reverb,pitch,shift,noise ... )",
                    min_value=0.0, max_value=1.0,
                    value=float(config.config['AUGMENTATION_PROB']),
                    key='augmentation_prob_input')
    st.number_input("Nivel de ruido", min_value=0.0, max_value=0.3,
                    value=float(config.config['NOISE_LEVEL']),
                    format="%.2f", key='noise_level_input')
    st.selectbox("Frecuencia de muestreo (sample rate)",
                 options=[16000, 22050, 44100, 48000],
                 index=[16000, 22050, 44100, 48000].index(config.config['SAMPLE_RATE']),
                 key='sample_rate_input')
    st.selectbox("Ref. en decibelios para normalizar espectrogramas",
                 options=[-10, -20, -30],
                 index=[-10, -20, -30].index(config.config['REF_DB']),
                 key='ref_db_input')
    st.selectbox("Normalización de audio (recomendable si los datos vienen de diferentes fuentes)",
                 options=['Si', 'No'],
                 index=0 if config.config['NORMALIZE_AUDIO'] else 1,
                 key='normalized_audio_input')


def save_and_exit_param_selection():
    # STFT
    config.config['STFT_WINDOW_LENGTH'] = int(st.session_state.get('stft_window_input'))
    config.config['STFT_HOP_LENGTH'] = int(st.session_state.get('stft_hop_length_input'))
    config.config['STFT_WINDOW_TYPE'] = st.session_state.get('stft_window_type_input')

    # Entrenamiento
    config.config['LEARNING_RATE'] = float(st.session_state.get('learning_rate_input'))
    config.config['BATCH_SIZE'] = int(st.session_state.get('batch_size_input'))
    config.config['MAX_EPOCHS'] = int(st.session_state.get('epochs_input'))
    config.config['EPOCH_LENGTH'] = int(st.session_state.get('epochs_length_input'))
    config.config['GRADIENT_CLIP'] = float(st.session_state.get('gradient_clip_input'))
    config.config['LEARNING_RATE_SCHEDULE'] = int(st.session_state.get('learning_rate_schedule_input'))
    config.config['EARLY_STOPPING_PATIENCE'] = int(st.session_state.get('early_stopping_patience_input'))
    config.config['WEIGHT_DECAY'] = float(st.session_state.get('weight_decay_input'))

    # Generación de mezclas
    config.config['COHERENT_PROB'] = float(st.session_state.get('coherent_prob_input'))

    # Modelo
    config.config['MODEL_NUM_CHANNELS'] = int(st.session_state.get('num_channels_input').split(',')[0])
    config.config['MODEL_DROPOUT'] = int(st.session_state.get('dropout_input'))
    config.config['MODEL_ACTIVATION'] = st.session_state.get('activation_func_input')
    config.config['EVALUATOR_FRAMES'] = int(st.session_state.get('evaluator_frames_input'))

    # FX y audio
    config.config['AUGMENTATION_PROB'] = float(st.session_state.get('augmentation_prob_input'))
    config.config['NOISE_LEVEL'] = float(st.session_state.get('noise_level_input'))
    config.config['SAMPLE_RATE'] = int(st.session_state.get('sample_rate_input'))
    config.config['REF_DB'] = int(st.session_state.get('ref_db_input'))
    config.config['NORMALIZE_AUDIO'] = st.session_state.get('normalized_audio_input') == 'Si'

    st.success("Parámetros actualizados correctamente.")
    st.session_state.page = 'main'
    st.rerun()
