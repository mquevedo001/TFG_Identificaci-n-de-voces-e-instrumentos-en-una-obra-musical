import streamlit as st
from pathlib import Path
import tempfile
import time
import shutil
from models.Mi_modelo.mask_inference import MaskInference
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from GUI.gui_aux_functions import (
    stft_param_selection,
    train_param_selection,
    mix_gen_param_selection,
    model_param_selection,
    fx_param_selection,
    save_and_exit_param_selection
)

from commons.metrics import run_training_and_capture_logs
from pipeline.evaluation import evaluation
from pipeline.deployment import deploy
from config import config
from config import loss_functions_help
from config import help_quotes
import nussl

# --- Inicialización de estado de sesión ---
if 'page' not in st.session_state:
    st.session_state.page = 'main'

for key in ['train', 'advanced_config']:
    if key not in st.session_state:
        st.session_state[key] = False

# --- Sidebar ---
st.sidebar.title("Funcionalidades para tu propio modelo")

num_sources_selection = st.sidebar.selectbox("Número de fuentes (stems)", options=[2, 4], index=0)

model_selected = st.sidebar.selectbox(
    "Modelos disponibles",
    options=['self', 'modelo_de_martin', 'modelo_de_usuario'],
    index=0
)

with st.sidebar.expander("Sobre el modelo", expanded=True):
    desc = help_quotes.model_descriptions.get(model_selected, "Sin descripción disponible para este modelo.")
    st.markdown(desc)

# Variables para base de datos y función de pérdida, con bloqueo condicional
if model_selected == 'modelo_de_martin':
    # Valores bloqueados
    database_selection = 'MUSDB18'
    loss_fn_selection = 'L2'

    st.sidebar.selectbox(
        "Bases de datos disponibles",
        options=['MUSDB18', 'Rock DB', 'HipHop DB'],
        index=0,
        disabled=True,
        key='db_locked'
    )

    st.sidebar.selectbox(
        "Función de pérdida",
        options=[
            'L1', 'L2','L1_freq', 'L2_freq', 'LOGL1_freq', 'LOGL2_freq',
            'LOG_mag', 'LOG_compressed_l2', 'MASK_L1', 'LPSA', 'LPSA_phase',
            'LMRS', 'L_MRS', 'Deep-feature', 'MSE', 'SDR','Deep-feature-EMD'
        ],
        index=1,  # Índice de 'L2'
        disabled=True,
        key='loss_locked'
    )

else:
    database_selection = st.sidebar.selectbox(
        "Bases de datos disponibles",
        options=['MUSDB18', 'Rock DB', 'HipHop DB']
    )

    loss_fn_selection = st.sidebar.selectbox(
        "Función de pérdida",
        options=[
            'L1', 'L2','L1_freq', 'L2_freq', 'LOGL1_freq', 'LOGL2_freq',
            'LOG_mag', 'LOG_compressed_l2', 'MASK_L1', 'LPSA', 'LPSA_phase',
            'LMRS', 'L_MRS', 'Deep-feature', 'MSE', 'SDR','Deep-feature-EMD'
        ],
        index=0
    )

with st.sidebar.expander("Sobre la función de pérdida", expanded=True):
    if loss_fn_selection in loss_functions_help.loss_key_map:
        loss_functions_help.loss_key_map[loss_fn_selection]()
    else:
        st.markdown("Sin descripción disponible para esta función de pérdida.")

with st.sidebar.expander("Sobre la base de datos", expanded=True):
    desc = help_quotes.database_descriptions.get(database_selection, "Sin descripción disponible para esta base de datos.")
    st.markdown(desc)

# --- Botones de acción ---
if st.sidebar.button('Cargar configuración seleccionada', use_container_width=True):
    # Guardar los valores bloqueados si es el modelo de Martin
    if model_selected == 'modelo_de_martin':
        config.config['MODEL_LOSS_FUNCTION'] = 'deep_feature_emd'
        config.config['MODEL_NUM_SOURCES'] = num_sources_selection
        # Si quieres guardar base de datos también, añádelo aquí si config lo soporta
        st.session_state.database_selection = 'MUSDB18'
        st.session_state.loss_fn_selection = 'Deep-feature-EMD'
    else:
        config.config['MODEL_LOSS_FUNCTION'] = loss_fn_selection
        config.config['MODEL_NUM_SOURCES'] = num_sources_selection
        st.session_state.database_selection = database_selection

    output_folder = Path(__file__).resolve().parent / 'checkpoints'
    output_folder.mkdir(parents=True, exist_ok=True)

    num_sources = config.config['MODEL_NUM_SOURCES']
    loss_fn = config.config['MODEL_LOSS_FUNCTION']

    if model_selected == 'self':
        model_path = output_folder / 'Modelo_base' / f'{loss_fn} checkpoints' / f'{num_sources}stems' / 'best.model.pth'
    elif model_selected == 'modelo_de_martin':
        model_path = output_folder / 'Mis_modelos' / f'{loss_fn} checkpoints' / f'{num_sources}stems' / 'best.model.pth'
    elif model_selected == 'modelo_de_usuario':
        model_path = output_folder / 'Modelos_de_usuario' / f'{loss_fn} checkpoints' / f'{num_sources}stems' / 'best.model.pth'

    st.session_state.loaded_model_name = model_selected
    st.session_state.loaded_model_path = str(model_path)

if st.sidebar.button("Entrenar modelo", use_container_width=True):
    st.session_state.page = 'train'
    st.rerun()

if st.sidebar.button("Configuración avanzada", use_container_width=True):
    st.session_state.page = 'advanced_config'
    st.rerun()

@st.cache_resource(show_spinner="Cargando modelo...")
def load_separator():
    audio_signal = nussl.AudioSignal()
    return nussl.separation.deep.DeepMaskEstimation(
        audio_signal,
        model_path=str(st.session_state.get('loaded_model_path', '')),
        device=config.config['DEVICE']
    )

# --- Página de entrenamiento ---
if st.session_state.page == 'train':
    st.title("Entrenamiento del modelo")

    if st.button('Entrenar', use_container_width=True):
        with st.spinner("Entrenando modelo..."):
            logs = run_training_and_capture_logs()
        st.success("Entrenamiento completado.")
        st.subheader("Registro de entrenamiento:")
        st.text_area("Logs", logs, height=400)

        if st.checkbox("¿Quieres descargar el modelo entrenado?"):
            model_path = Path(st.session_state.loaded_model_path)
            temp_path = Path(__file__).parent / 'GUI' / 'temp_model.pth'
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(model_path, temp_path)

            with open(temp_path, 'rb') as f:
                st.download_button(
                    label="Descargar modelo entrenado (.pth)",
                    data=f,
                    file_name='best.model.pth',
                    mime='application/octet-stream'
                )

    if st.button("Volver", use_container_width=True):
        st.session_state.page = 'main'
        st.rerun()
    st.stop()

# --- Página de configuración avanzada ---
if st.session_state.page == 'advanced_config':
    st.title("Configuración avanzada para tu modelo")
    st.markdown("Modifica los parámetros de entrenamiento:")
    col1_advanced, col2_advanced = st.columns([2, 2])

    with col1_advanced:
        stft_param_selection()
        train_param_selection()
        mix_gen_param_selection()

    with col2_advanced:
        model_param_selection()
        fx_param_selection()

    if st.button("Guardar y volver"):
        save_and_exit_param_selection()
    if st.button("Cancelar"):
        st.session_state.page = 'main'
        st.rerun()

# --- Página principal ---
st.title('Separación de música por stems')

loaded_model = st.session_state.get('loaded_model_name', 'Ninguno')
st.subheader(f"Modelo cargado actualmente: **{loaded_model}**")

col1, col2 = st.columns([2, 2])
with col1:
    uploaded_file = st.file_uploader("Sube un archivo de audio", type=["wav", "mp3"], accept_multiple_files=False)

    if uploaded_file:
        st.audio(uploaded_file, format='audio/wav')

        if st.button('Separar (Test)', use_container_width=True):
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_file_path = tmp_file.name

            st.info("Iniciando separación de audio...")
            progress_bar = st.progress(0, text="Procesando...")

            with st.spinner("Separando audio con el modelo..."):
                for percent in range(0, 100, 20):
                    time.sleep(0.3)
                    progress_bar.progress(percent + 1, text=f"Procesando... {percent + 1}%")

                stem_paths, sources_dict = deploy(output_dir=Path(__file__).parent / 'checkpoints', audio_path=tmp_file_path)

            progress_bar.progress(100, text="¡Completado!")

            st.success("Separación completada. Revisa la carpeta de salida.")
            st.subheader('Resultados:')

            for source, path in sources_dict.items():
                st.markdown(f'*{source}*')
                st.audio(path)
                with open(path, 'rb') as f:
                    st.download_button(f"⬇️ Descargar {source}", data=f, file_name=f"{source}.wav", mime="audio/wav")

                # Mostrar waveform
                y, sr = librosa.load(path, sr=None)
                fig_wave, ax = plt.subplots()
                librosa.display.waveshow(y, sr=sr, ax=ax)
                ax.set(title=f"Waveform: {source}")
                st.pyplot(fig_wave)

                # Mostrar espectrograma
                S = librosa.stft(y)
                S_dB = librosa.amplitude_to_db(np.abs(S), ref=np.max)
                fig_spec, ax = plt.subplots()
                librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='hz', ax=ax)
                ax.set(title=f"Espectrograma: {source}")
                st.pyplot(fig_spec)

        if st.button('Evaluar (Eval)', use_container_width=True):
            separator = load_separator()
            evaluation(frames=5, separator=separator)
            st.success("Evaluación finalizada.")
    else:
        st.info("Sube un archivo para interactuar con el modelo")

with col2:
    st.info("Comparativa del modelo (en desarrollo)")
