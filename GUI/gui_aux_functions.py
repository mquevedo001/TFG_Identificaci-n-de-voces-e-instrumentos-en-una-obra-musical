import io
import contextlib
import logging
import streamlit as st
from config import config
from pathlib import Path
from commons.experiment_utils import MAIN_LOSSES,EXPERIMENTAL_LOSSES,ADVANCED_LOSSES
from config.config import PROJECT_ROOT

CHECKPOINTS_ROOT = PROJECT_ROOT / "checkpoints" / "Mis_modelos_v2"

# ============================================================
# UTILIDADES: AYUDA DE PARÁMETROS
# ============================================================

def get_parameter_help_text() -> str:
    """
    Devuelve una explicación resumida de los parámetros configurables
    desde la interfaz avanzada.
    """
    return """PARÁMETROS STFT

STFT_WINDOW_LENGTH:
Tamaño de la ventana usada en la STFT. Define cuántas muestras de audio se analizan en cada frame temporal. Una ventana mayor ofrece más resolución en frecuencia, pero menor resolución temporal.

STFT_HOP_LENGTH:
Número de muestras que avanza la ventana entre frames consecutivos. Un hop menor produce más solapamiento entre ventanas y una representación temporal más densa.

STFT_WINDOW_TYPE:
Tipo de ventana aplicada antes de la transformada de Fourier. En este proyecto se utiliza habitualmente sqrt_hann, que ayuda a reducir discontinuidades entre ventanas.

SAMPLE_RATE:
Frecuencia de muestreo del audio. Indica cuántas muestras por segundo se utilizan para representar la señal.


PARÁMETROS DEL MODELO

MODEL_NUM_SOURCES:
Número de fuentes o stems que el modelo debe separar. En este trabajo se usan configuraciones de 2 stems y 4 stems.

MODEL_NUM_CHANNELS:
Número de canales de audio procesados por el modelo. En este proyecto se trabaja normalmente en mono, por lo que el valor habitual es 1.

MODEL_HIDDEN_SIZE:
Tamaño del estado oculto de la LSTM. Controla la capacidad interna de la red recurrente. Un valor mayor aumenta la capacidad del modelo, pero también el coste computacional.

MODEL_NUM_LAYERS:
Número de capas recurrentes apiladas en la LSTM. Más capas permiten aprender representaciones más complejas, aunque pueden dificultar el entrenamiento.

MODEL_BIDIRECTIONAL:
Indica si la LSTM procesa la secuencia en ambas direcciones temporales. Si está activado, el modelo utiliza información de frames anteriores y posteriores.

MODEL_DROPOUT:
Técnica de regularización que desactiva aleatoriamente parte de las conexiones durante el entrenamiento para reducir sobreajuste.

MODEL_ACTIVATION:
Función de activación usada en la salida del modelo. En inferencia de máscaras se utiliza habitualmente sigmoid para limitar los valores de máscara al intervalo [0, 1].


PARÁMETROS DE ENTRENAMIENTO

MODEL_LOSS_FUNCTION:
Función de pérdida empleada para comparar la salida del modelo con la referencia. Permite estudiar cómo afectan distintas formulaciones del error al rendimiento de separación.

BATCH_SIZE:
Número de ejemplos procesados antes de actualizar los pesos del modelo. Un batch mayor puede estabilizar el entrenamiento, pero requiere más memoria.

LEARNING_RATE:
Tasa de aprendizaje del optimizador. Controla cuánto se modifican los pesos del modelo en cada actualización.

WEIGHT_DECAY:
Regularización aplicada sobre los pesos del modelo. Ayuda a reducir el sobreajuste penalizando pesos excesivamente grandes.

MAX_EPOCHS:
Número máximo de épocas de entrenamiento. Una época corresponde a recorrer una vez el conjunto de entrenamiento.

PATIENCE:
Número de épocas sin mejora antes de detener el entrenamiento de forma temprana mediante early stopping.

GRAD_CLIP:
Valor máximo usado para recortar gradientes. Ayuda a evitar explosiones de gradiente durante el entrenamiento recurrente.

DEVICE:
Dispositivo utilizado para entrenar o ejecutar el modelo. Puede ser cpu o cuda, dependiendo de si se dispone de GPU.


PARÁMETROS DE GENERACIÓN DE MEZCLAS

MIXTURE_DURATION:
Duración de los fragmentos de audio usados durante el entrenamiento.

NUM_WORKERS:
Número de procesos utilizados para cargar datos. Un valor mayor puede acelerar la carga, aunque depende del equipo.

SHUFFLE:
Indica si los ejemplos de entrenamiento se barajan antes de cada época. Esto ayuda a mejorar la generalización del modelo.


PARÁMETROS DE INFERENCIA Y EVALUACIÓN

CHECKPOINT_PATH:
Ruta del checkpoint que se desea cargar para inferencia o evaluación.

MODEL_PATH:
Ruta del modelo seleccionado desde la interfaz.

EVALUATION_FRAMES:
Número de frames o fragmentos utilizados durante la evaluación desde la GUI.

OUTPUT_DIR:
Carpeta donde se guardan los audios separados, gráficos y logs generados desde la interfaz.
"""

def get_loss_group_from_name(loss_name: str) -> str:
    loss_name = loss_name.lower()

    if loss_name in MAIN_LOSSES:
        return "main"

    if loss_name in EXPERIMENTAL_LOSSES:
        return "experimental"

    return "advanced"

def get_checkpoint_folder(loss_name: str, num_sources: int) -> Path:
    """
    Devuelve la carpeta donde deberían estar los checkpoints
    para una loss y número de stems concretos.
    """
    loss_name = str(loss_name).lower()
    group = get_loss_group_from_name(loss_name)

    return (
        CHECKPOINTS_ROOT
        / group
        / f"{loss_name} checkpoints"
        / f"{num_sources}stems"
    )


def list_checkpoints(loss_name: str, num_sources: int):
    """
    Busca checkpoints .pt y .pth dentro de la carpeta correspondiente.
    Usa rglob para encontrar también checkpoints si están en subcarpetas.
    """
    checkpoint_folder = get_checkpoint_folder(loss_name, num_sources)

    if not checkpoint_folder.exists():
        return []

    candidates = []
    candidates.extend(checkpoint_folder.rglob("*.pt"))
    candidates.extend(checkpoint_folder.rglob("*.pth"))

    # Ordenar primero los que contienen 'best', luego por fecha reciente
    candidates = sorted(
        candidates,
        key=lambda p: (
            "best" not in p.name.lower(),
            -p.stat().st_mtime
        )
    )

    return candidates

   
class _StreamToWidget(io.StringIO):
    """Un stream que vuelca todo a un placeholder de Streamlit en vivo."""
    def __init__(self, placeholder):
        super().__init__()
        self.placeholder = placeholder
    def write(self, s):
        super().write(s)
        # Muestra todo el buffer acumulado (puedes cambiar a .code si prefieres monospace)
        self.placeholder.text(self.getvalue())
    def flush(self):
        pass

@contextlib.contextmanager
def live_console():
    """Contexto que captura stdout, stderr y logging hacia un panel de Streamlit."""
    placeholder = st.empty()
    stream = _StreamToWidget(placeholder)

    # Redirige stdout/stderr
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        # Redirige logging
        root_logger = logging.getLogger()
        root_level = root_logger.level
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.INFO)
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)
        try:
            yield stream  # por si quieres obtener el texto al final
        finally:
            # Limpieza
            root_logger.removeHandler(handler)
            root_logger.setLevel(root_level)


def stft_param_selection():


    hop_options = [window_length // 2 , window_length // 4]
    current_hop = int(config.config["STFT_HOP_LENGTH"])
    hop_index = (hop_options.index(current_hop) if current_hop in hop_options else 1)
    st.subheader('Parámetros de la formación de transformadas de Fourier')

    window_length = st.selectbox("Tamaño de la ventana STFT", options=[512, 1024, 2048, 4096],index=[512, 1024, 2048, 4096].index(config.config.get('STFT_WINDOW_LENGTH')),key='stft_window_input')
    st.selectbox("Desplazamiento entre ventanas STFT",options=hop_options,index=hop_index,key='stft_hop_length_input')
    st.selectbox("Tipo de ventana utilizada para la STFT",options=["sqrt_hann", "hann", "hamming", "blackman", "bartlett"],index=0,key='stft_window_type_input')
                 
def train_param_selection():

    st.subheader("Parámetros de entrenamiento")

    st.number_input("Learning Rate", value=float(config.config.get('LEARNING_RATE', 0.001)),format="%.5f", key='learning_rate_input')                 
    st.number_input("Batch Size", min_value=1, value=int(config.config.get('BATCH_SIZE', 16)),key='batch_size_input')                 
    st.number_input("Épocas", min_value=1, value=int(config.config.get('MAX_EPOCHS', 50)),key='epochs_input')                 
    st.number_input("Tamaño de las épocas", min_value=10, value=int(config.config['EPOCH_LENGTH']),key='epochs_length_input')                   
    st.number_input("Gradient clip", min_value=0.0, value=float(config.config['GRADIENT_CLIP']),format="%.2f", key='gradient_clip_input')            
    st.number_input("Weight decay", min_value=0.0, value=float(config.config['WEIGHT_DECAY']),format="%.6f", key='weight_decay_input')
                    


def mix_gen_param_selection():

    st.subheader("Parámetros de generación de mezclas")
    st.number_input("Coherent prob", min_value=0.0, max_value=1.0,value=float(config.config['COHERENT_PROB']), key='coherent_prob_input')
                    


def model_param_selection():

    st.subheader('Parámetros intrínsecos modelo')
    st.selectbox("Número de canáles de entrada (Mono/Stereo)",options=[1,2], key='num_channels_input')

    st.number_input("Dropout",min_value=0.0,max_value=1.0,step=0.05,value=float(config.config['MODEL_DROPOUT']),key='dropout_input')
    st.number_input("Early stopping patience",min_value=1,value=int(config.config["EARLY_STOPPING_PATIENCE"]),key="early_stopping_patience_input")

    st.selectbox("Funcion de activación para la capa de salida",options=['softmax', 'relu', 'sigmoid', 'leaky_relu'],index=['softmax', 'relu', 'sigmoid', 'leaky_relu'].index(config.config['MODEL_ACTIVATION']),key='activation_func_input')            
    st.selectbox("Numero de frames para la evaluación del modelo",options=[0, 5, 50, 100, 500, 1000],index=[0, 5, 50, 100, 500, 1000].index(int(config.config.get("EVALUATOR_FRAMES",10))),key='evaluator_frames_input')
                 
                 
def fx_param_selection():

    st.subheader('Parámetros de efectos para entrenamiento y de audio')

    st.number_input("Prob. de aplicar efectos (reverb,pitch,shift,noise ... )",min_value=0.0, max_value=1.0,value=float(config.config['AUGMENTATION_PROB']),key='augmentation_prob_input')                                                       
    st.number_input("Nivel de ruido", min_value=0.0, max_value=0.3,value=float(config.config['NOISE_LEVEL']),format="%.2f", key='noise_level_input')
                                      
    st.selectbox("Frecuencia de muestreo (sample rate)",options=[16000, 22050, 44100, 48000],index=[16000, 22050, 44100, 48000].index(config.config['SAMPLE_RATE']),key='sample_rate_input')                                      
    st.selectbox("Ref. en decibelios para normalizar espectrogramas",options=[-10, -20, -30],index=[-10, -20, -30].index(config.config['REF_DB']),key='ref_db_input')         
    st.selectbox("Normalización de audio (recomendable si los datos vienen de diferentes fuentes)",options=['Si', 'No'],index=0 if config.config['NORMALIZE_AUDIO'] else 1,key='normalized_audio_input')
                 
                 
                 


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
    #config.config['LEARNING_RATE_SCHEDULE'] = int(st.session_state.get('learning_rate_schedule_input'))
    config.config['EARLY_STOPPING_PATIENCE'] = int(st.session_state.get('early_stopping_patience_input'))
    config.config['WEIGHT_DECAY'] = float(st.session_state.get('weight_decay_input'))

    # Generación de mezclas
    config.config['COHERENT_PROB'] = float(st.session_state.get('coherent_prob_input'))

    # Modelo
    config.config['MODEL_NUM_CHANNELS'] = int(st.session_state.get('num_channels_input'))
    config.config['MODEL_DROPOUT'] = float(st.session_state.get('dropout_input'))
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
