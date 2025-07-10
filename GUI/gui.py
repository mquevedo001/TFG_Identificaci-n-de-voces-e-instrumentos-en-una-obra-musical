import streamlit as st
import sys
from pathlib import Path

# Añadir raíz del proyecto al sys.path
root_dir = Path(__file__).resolve().parent
sys.path.append(str(root_dir))

# Imports locales
from pipeline.train import training
from pipeline.evaluation import evaluation
from pipeline.deployment import deploy
from config import config
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
        input_path = output_folder / uploaded_file.name

        with open(input_path, 'wb') as f:
            f.write(uploaded_file.getbuffer().name)

        stem_outputs = deploy(output_dir=output_folder, audio_path= str(input_path))

        st.success("Separación completada. Revisa la carpeta de salida.")
        st.subheader('Resultados: ')
        for stem_name,stem_path in stem_outputs:
            st.markdown(f"**{stem_name.capitalize()}**")
            st.audio(stem_path)
        st.audio(uploaded_file)

    if st.button('Evaluar modelo',use_container_width = True):
        separator = load_separator()
        evaluation(frames=5, separator=separator)
        st.success("Evaluación finalizada.")
else:
    st.info("Sube un archivo para interactuar con el modelo")

#TODO
#Añadir parámetros para jugar con el modelo
#Arreglar display de output del modelo
