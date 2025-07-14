import streamlit as st
import sys
from pathlib import Path
import tempfile
import time

# Añadir raíz del proyecto al sys.path
root_dir = Path(__file__).resolve().parent
sys.path.append(str(root_dir))

# Imports locales
from commons.metrics import run_training_and_capture_logs
from pipeline.evaluation import evaluation
from pipeline.deployment import deploy
from config import config
from config import loss_functions_help  # aquí está el archivo con las funciones que hacen st.latex
import nussl

# --- Inicialización de estado de sesión ---
if 'page' not in st.session_state:
    st.session_state.page = 'main'

# --- Sidebar ---
st.sidebar.title("Funcionalidades para tu propio modelo")

num_sources = st.sidebar.selectbox("Número de fuentes (stems)", options=[2, 4], index=0)

loss_fn = st.sidebar.selectbox("Función de pérdida", options=[
    'L1', 'L2','L1_freq', 'L2_freq', 'LOGL1_freq', 'LOGL2_freq',
    'LOG_mag', 'LOG_compressed_l2', 'MASK_L1', 'LPSA', 'LPSA_phase',
    'LMRS', 'L_MRS', 'Deep-feature', 'MSE', 'SDR'
], index=0)

# Mostrar descripción con fórmula renderizada usando funciones en loss_functions_help
with st.sidebar.expander("Descripción de la función de pérdida", expanded=False):
    if loss_fn in loss_functions_help.loss_key_map:
        loss_functions_help.loss_key_map[loss_fn]()  # Llama a la función que hace st.latex y st.markdown
    else:
        st.markdown("Sin descripción disponible para esta función de pérdida.")

model_selected = st.sidebar.selectbox("Modelos disponibles", options=['self', 'modelo_de_martin', 'modelo_de_usuario'], index=0)
database_selection = st.sidebar.selectbox("Bases de datos disponibles", options=['MUSDB18', 'Rock DB', 'HipHop DB'])

# Aplicar a config global
config.config['MODEL_LOSS_FUNCTION'] = loss_fn
config.config['MODEL_NUM_SOURCES'] = num_sources

output_folder = root_dir / 'checkpoints'

model_path = output_folder / 'Mis_modelos' / f'{loss_fn} checkpoints' / f'{num_sources}stems' / 'best.model.pth'
user_model_path = output_folder / 'Modelos_de_usuario' / f'{loss_fn} checkpoints' / f'{num_sources}stems' / 'best.model.pth'
base_model_path = output_folder / 'Modelo_base' / f'{loss_fn} checkpoints' / f'{num_sources}stems' / 'best.model.pth'

output_folder.mkdir(parents=True, exist_ok=True)

@st.cache_resource(show_spinner="Cargando modelo...")
def load_separator():
    audio_signal = nussl.AudioSignal()
    return nussl.separation.deep.DeepMaskEstimation(
        audio_signal,
        model_path=str(model_path.resolve()),
        device=config.config['DEVICE']
    )

# --- Página de entrenamiento ---
if st.session_state.page == 'train':
    st.title("Entrenamiento del modelo")

    if st.button('Entrenar', use_container_width=True):
        st.session_state.current_page = "train"

        with st.spinner("Entrenando modelo..."):
            logs = run_training_and_capture_logs()

        st.success("Entrenamiento completado.")
        st.subheader("Registro de entrenamiento:")
        st.text_area("Logs", logs, height=400)

    if st.button("Volver", use_container_width=True):
        st.session_state.page = 'main'
    st.stop()

# --- Página principal ---
st.title('Separación de música por stems')

col1, col2 = st.columns([2, 2])
with col1:
    uploaded_file = st.file_uploader("Sube un archivo de audio", type=["wav", "mp3"], accept_multiple_files=False)

    if uploaded_file:
        st.audio(uploaded_file, format='audio/wav')

        if st.button('Entrenar (Train)', use_container_width=True, help='cuidadoooo'):
            st.session_state.page = 'train'
            st.rerun()

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

                stem_paths, sources_dict = deploy(output_dir=output_folder, audio_path=tmp_file_path)

            progress_bar.progress(100, text="¡Completado!")

            st.success("Separación completada. Revisa la carpeta de salida.")
            st.subheader('Resultados:')

            vocals_path = Path('.') / 'Results' / 'Separation' / 'stems' / 'vocals.wav'
            accompaniment_path = Path('.') / 'Results' / 'Separation' / 'stems' / 'accompaniment.wav'

            st.markdown('*Vocales*')
            st.audio(vocals_path)
            st.markdown('*Acompañamiento*')
            st.audio(accompaniment_path)

        if st.button('Evaluar (Eval)', use_container_width=True):
            separator = load_separator()
            evaluation(frames=5, separator=separator)
            st.success("Evaluación finalizada.")

    else:
        st.info("Sube un archivo para interactuar con el modelo")

with col2:
    st.info("Gráficos del modelo (en desarrollo)")

# --- Sidebar extra: cargar configuración ---
if st.sidebar.button('Cargar configuración seleccionada', use_container_width=True):
    if model_selected == 'self':
        model_path = base_model_path
    elif model_selected == 'modelo_de_martin':
        model_path = model_path
    elif model_selected == 'modelo_de_usuario':
        model_path = user_model_path
