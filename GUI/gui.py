import streamlit as st
import sys
from pathlib import Path
import tempfile

# Añadir raíz del proyecto al sys.path
root_dir = Path(__file__).resolve().parent
sys.path.append(str(root_dir))

# Imports locales
from pipeline.train import training
from pipeline.evaluation import evaluation
from pipeline.deployment import deploy
from config import config
import nussl

# --- Sidebar ---
st.sidebar.title("Funcionalidades para tu propio modelo")

# Parámetros del modelo desde la sidebar
num_sources = st.sidebar.selectbox("Número de fuentes (stems)", options=[2, 4], index=0)
loss_fn = st.sidebar.selectbox("Función de pérdida", options=['L1', 'MSE', 'SDR'], index=0)

st.sidebar.title('Modelos')

model_selected = st.sidebar.selectbox("Modelos disponibles" , options = ['self','modelo de martin','modelo original ISMIR'],index = 0)

database_selection = st.sidebar.selectbox("Bases de datos disponibles",options = ['MUSDB18' , 'Rock DB' , 'HipHop DB'])

# Aplicar a config global
config.config['MODEL_LOSS_FUNCTION'] = loss_fn
config.config['MODEL_NUM_SOURCES'] = num_sources

# Actualizar ruta del modelo en función de la config seleccionada
output_folder = root_dir / 'checkpoints'
model_path = output_folder / 'Mi_modelo' /f'{loss_fn} checkpoints' / f'{num_sources}stems' / 'best.model.pth'

output_folder.mkdir(parents=True, exist_ok=True)

# --- Parámetros globales ---
loss_fn = config.config['MODEL_LOSS_FUNCTION']
num_sources = config.config['MODEL_NUM_SOURCES']


# Asegurar que el directorio de salida existe
output_folder.mkdir(parents=True, exist_ok=True)

@st.cache_resource(show_spinner="Cargando modelo...")
def load_separator():
    """Carga el modelo solo una vez"""
    audio_signal = nussl.AudioSignal()
    return nussl.separation.deep.DeepMaskEstimation(
        audio_signal,
        model_path=str(model_path.resolve()),
        device=config['DEVICE']
    )

# --- UI ---
st.title('Separación de música por stems')

uploaded_file = st.file_uploader("Sube un archivo de audio", type=["wav", "mp3"], accept_multiple_files=False)

if uploaded_file:
    st.audio(uploaded_file, format='audio/wav')

    if st.button('Entrenar (Train)', use_container_width=True):
        training()
        st.success("Entrenamiento completado.")

    if st.button('Separar (Test)', use_container_width=True):
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_file_path = tmp_file.name

        st.info("Iniciando separación de audio...")

        # Mostrar barra de progreso (simulada)
        progress_bar = st.progress(0, text="Procesando...")

        with st.spinner("Separando audio con el modelo..."):
            # Simular carga progresiva (opcional)
            import time

            for percent in range(0, 100, 20):
                time.sleep(0.3)
                progress_bar.progress(percent + 1, text=f"Procesando... {percent + 1}%")

            # Aquí corre la separación real
            stem_paths, sources_dict = deploy(output_dir=output_folder, audio_path=tmp_file_path)

        progress_bar.progress(100, text="¡Completado!")

        st.success("Separación completada. Revisa la carpeta de salida.")
        st.subheader('Resultados:')

        vocals_path = Path('.') / 'Results' / 'stems' / 'vocals.wav'
        accompaniment_path = Path('.') / 'Results' / 'stems' / 'accompaniment.wav'

        st.markdown('*Vocales*')
        st.audio(vocals_path)
        st.markdown('*Acompañamiento*')
        st.audio(accompaniment_path)

    if st.button('Evaluar (Eval)', use_container_width=True):
        separator = load_separator()
        evaluation(frames=5, separator=separator)
        st.success("Evaluación finalizada.")

    if st.sidebar.button('Utilizar el modelo base',use_container_width=True):

        model_path = output_folder / 'Modelo_base' /f'{loss_fn} checkpoints' / f'{num_sources}stems' / 'best.model.pth'


    if st.sidebar.button('Utilizar tu modelo',use_container_width=True):

        model_path = output_folder / 'Modelos_de_usuario' /f'{loss_fn} checkpoints' / f'{num_sources}stems' / 'best.model.pth'


    if st.sidebar.button('Utilizar mi modelo',use_container_width=True):

        model_path = output_folder / 'Mi_modelo' /f'{loss_fn} checkpoints' / f'{num_sources}stems' / 'best.model.pth'

else:
    st.info("Sube un archivo para interactuar con el modelo")

# TODO:
# - Añadir parámetros para experimentar con el modelo
# - Mostrar mejor los resultados de separación
# - Crear las diferentes BDs y permitir la descarga y subida de modelos
# - Añadir a la GUI apartado de gráficos