import streamlit as st
import sys
from pathlib import Path

# Añadir raíz del proyecto al sys.path
root_dir = Path(__file__).resolve().parent
sys.path.append(str(root_dir))

# Imports locales
from training.train import training
from training.evaluation import evaluation
from deployment import deploy
from utils.config import config
import nussl

# --- Parámetros globales ---
output_folder = root_dir / 'checkpoints'
model_path = output_folder / 'best.model.pth'


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
    st.audio(uploaded_file)

    if st.button('Entrenar modelo',use_container_width = True):
        training()
        st.success("Entrenamiento completado.")

    if st.button('Separar (Test)',use_container_width = True):
        audio_signal = deploy(output_dir=output_folder, audio_path=uploaded_file.name)
        st.success("Separación completada. Revisa la carpeta de salida.")
        st.audio(uploaded_file)

    if st.button('Evaluar modelo',use_container_width = True):
        separator = load_separator()
        evaluation(frames=5, separator=separator)
        st.success("Evaluación finalizada.")
else:
    st.info("Sube un archivo para interactuar con el modelo")
