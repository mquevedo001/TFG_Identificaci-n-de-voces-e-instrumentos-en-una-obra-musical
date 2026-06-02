import streamlit as st
import sys
sys.path.insert(0,"C:\\Users\\rdpuser\\TFG_Identificaci-n-de-voces-e-instrumentos-en-una-obra-musical")
from pathlib import Path
import tempfile
import time
import shutil
from models.Mi_modelo import mask_inference

from gui_aux_functions import live_console
import torch
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
# Sidebar — después de “Cargar configuración seleccionada”
if st.sidebar.button('Cargar configuración seleccionada', use_container_width=True):
    # tipos y config
    config.config['MODEL_NUM_SOURCES'] = int(num_sources_selection)
    config.config['MODEL_LOSS_FUNCTION'] = 'deep_feature_emd' if model_selected == 'modelo_de_martin' else loss_fn_selection

    output_folder = Path(__file__).resolve().parent / 'checkpoints'
    output_folder.mkdir(parents=True, exist_ok=True)

    if model_selected == 'self':
        base = output_folder / 'Modelo_base'
    elif model_selected == 'modelo_de_martin':
        base = output_folder / 'Mis_modelos'
    else:
        base = output_folder / 'Modelos_de_usuario'

    model_path = base / f"{loss_fn_selection.lower()} checkpoints" / f"{num_sources_selection}stems" / 'checkpoints' / 'best.model.pth'

    model_path = (
            Path(
                "/home/martin/PycharmProjects/TFG_Identificaci-n-de-voces-e-instrumentos-en-una-obra-musical/checkpoints/Mis_modelos")
            / f"{config.config['MODEL_LOSS_FUNCTION']} checkpoints"
            / f"{config.config['MODEL_NUM_SOURCES']}stems"
            / "checkpoints"
            / "best.model.pth"
    )

    st.session_state.loaded_model_name = model_selected
    st.session_state.loaded_model_path = str(model_path)

    if not model_path.exists():
        st.warning(
            f"No se encontró el checkpoint en:\n`{model_path}`.\n"
            "Puedes entrenar uno nuevo desde la pestaña *Entrenamiento*."
        )
    else:
        st.success("Configuración cargada.")


if st.sidebar.button("Entrenar modelo", use_container_width=True):
    st.session_state.page = 'train'
    st.rerun()

if st.sidebar.button("Configuración avanzada", use_container_width=True):
    st.session_state.page = 'advanced_config'
    st.rerun()

@st.cache_resource(show_spinner="Cargando modelo...")
def load_separator():
    nf = config.config['STFT_WINDOW_LENGTH'] // 2 + 1

    model = MaskInference.build(
        nf,
        num_audio_channels=config.config['MODEL_NUM_CHANNELS'],
        hidden_size=config.config['MODEL_HIDDEN_SIZE'],
        num_layers=config.config['MODEL_NUM_LAYERS'],
        bidirectional=config.config['MODEL_BIDIRECTIONAL'],
        dropout=config.config['MODEL_DROPOUT'],
        num_sources=num_sources_selection,
        activation=config.config['MODEL_ACTIVATION']
    )

    # Cargar checkpoint nussl (puede venir con 'state_dict', 'config', 'metadata')
    checkpoint = torch.load(model_path, map_location=config.config['DEVICE'])

    # Extraer el state_dict correcto
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint  # por si fuese un state_dict plano

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print('[WARN] Claves faltantes al cargar:', missing)
    if unexpected:
        print('[WARN] Claves inesperadas al cargar:', unexpected)

    return model

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

col1, col2 ,col3= st.columns([2, 2,2])
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
    st.subheader("Consola")
    live_console()

with col3:

    st.info("Comparativa del modelo (en desarrollo)")

