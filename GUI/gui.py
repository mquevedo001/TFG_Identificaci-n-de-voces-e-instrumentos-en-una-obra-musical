# ============================================================
# GUI - Separación de música por stems
# ============================================================
# Este archivo implementa:
#   - Selección de modelo, pérdida, número de stems y checkpoint.
#   - Carga robusta de checkpoints .pt/.pth.
#   - Enrutamiento correcto de salidas de audio y gráficos.
#   - Visualización de audio, waveform y espectrograma.
#   - Acceso a entrenamiento, configuración avanzada y evaluación.
# ============================================================

import sys
import time
import shutil
import tempfile
import inspect
import html
from pathlib import Path
from collections import OrderedDict

import streamlit as st
import torch
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

import contextlib
import traceback
from datetime import datetime
from commons.experiment_utils import checkpoint_dir


# ============================================================
# PATHS DEL PROYECTO
# ============================================================

GUI_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = GUI_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS DEL PROYECTO
# ============================================================

from models.Mi_modelo.mask_inference import MaskInference
from GUI.gui_aux_functions import *
from GUI.gui_aux_functions import live_console
from GUI.gui_aux_functions import get_parameter_help_text

from commons.metrics import run_training_and_capture_logs
from commons.runtime import resolve_device
from pipeline.evaluation import evaluation
from pipeline.deployment import deploy
from config import config
from config import loss_functions_help
from config import help_quotes

try:
    from commons.experiment_utils import get_loss_group
except Exception:
    get_loss_group = None


# ============================================================
# CONSTANTES Y OPCIONES
# ============================================================

CHECKPOINTS_ROOT = PROJECT_ROOT / "checkpoints"
CHECKPOINTS_V2_ROOT = CHECKPOINTS_ROOT / "Mis_modelos_v2"
GUI_RESULTS_ROOT = PROJECT_ROOT / "resultados_gui"

GUI_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

LOSS_OPTIONS = [
    "L1",
    "L2",
    "L1_freq",
    "L2_freq",
    "LOGL1_freq",
    "LOGL2_freq",
    "LOG_mag",
    "LOG_compressed_l2",
    "MASK_L1",
    "LPSA",
    "LPSA_phase",
    "LMRS",
    "L_MRS",
    "Deep-feature",
    "MSE",
    "SDR",
    "Deep-feature-EMD",
]

LOSS_DISPLAY_TO_KEY = {
    "L1": "l1",
    "L2": "l2",
    "L1_freq": "l1_freq",
    "L2_freq": "l2_freq",
    "LOGL1_freq": "logl1",
    "LOGL2_freq": "logl2",
    "LOG_mag": "log_mag",
    "LOG_compressed_l2": "log_compressed_l2",
    "MASK_L1": "mask_l1",
    "LPSA": "lpsa",
    "LPSA_phase": "lpsa_phase",
    "LMRS": "lmrs",
    "L_MRS": "l_mrs",
    "Deep-feature": "deep_feature",
    "Deep-feature-EMD": "deep_feature_emd",
    "MSE": "l2",
    "SDR": "sdr",
}

MAIN_LOSSES_FALLBACK = {
    "l1",
    "l2",
    "l1_freq",
    "l2_freq",
    "logl1",
    "logl2",
    "log_mag",
    "log_compressed_l2",
    "lpsa",
    "mask_l1",
}

EXPERIMENTAL_LOSSES_FALLBACK = {
    "deep_feature",
    "deep_feature_emd",
}

ADVANCED_LOSSES_FALLBACK = {
    "l_mrs",
    "lmrs",
    "lpsa_phase",
    "sdr",
}


# ============================================================
# UTILIDADES: CONSOLA / LOGS EN STREAMLIT
# ============================================================

MAX_CONSOLE_LINES = 500


def init_gui_console():
    if "console_lines" not in st.session_state:
        st.session_state.console_lines = []


def append_console(text: str):
    """
    Añade texto a la consola de la GUI.
    """
    init_gui_console()

    if text is None:
        return

    text = str(text).replace("\r\n", "\n")

    for line in text.splitlines():
        if line.strip():
            st.session_state.console_lines.append(line)

    st.session_state.console_lines = st.session_state.console_lines[-MAX_CONSOLE_LINES:]


def clear_console():
    st.session_state.console_lines = []


def get_console_text():
    init_gui_console()

    if not st.session_state.console_lines:
        return "Consola lista. Ejecuta una separación, evaluación o entrenamiento para ver logs."

    return "\n".join(st.session_state.console_lines)


def render_console(placeholder=None, height: int = 260):
    """
    Renderiza la consola con altura fija y scroll vertical.
    """
    if placeholder is None:
        placeholder = st.empty()

    console_text = html.escape(get_console_text())

    placeholder.markdown(
        f"""
        <div style="
            height: {height}px;
            overflow-y: auto;
            overflow-x: auto;
            white-space: pre-wrap;
            font-family: Consolas, 'Courier New', monospace;
            font-size: 0.85rem;
            line-height: 1.35;
            background-color: rgba(128, 128, 128, 0.12);
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 0.5rem;
            padding: 0.75rem;
        ">
{console_text}
        </div>
        """,
        unsafe_allow_html=True,
    )

    return placeholder


class StreamlitConsoleWriter:
    """
    Redirige stdout/stderr a Streamlit y también conserva salida en terminal.
    """
    def __init__(self, original_stream, placeholder=None):
        self.original_stream = original_stream
        self.placeholder = placeholder
        self.last_update = 0.0

    def write(self, text):
        if self.original_stream is not None:
            self.original_stream.write(text)
            self.original_stream.flush()

        if text and text.strip():
            append_console(text)

            now = time.time()
            if self.placeholder is not None and (now - self.last_update) > 0.2:
                render_console(self.placeholder, height=200)
                self.last_update = now

    def flush(self):
        if self.original_stream is not None:
            self.original_stream.flush()


@contextlib.contextmanager
def capture_logs_to_gui(placeholder=None, title=None):
    """
    Captura prints y errores durante un bloque de código.
    """
    if title:
        timestamp = datetime.now().strftime("%H:%M:%S")
        append_console(f"[{timestamp}] {title}")
        if placeholder is not None:
            render_console(placeholder, height=200)

    old_stdout = sys.stdout
    old_stderr = sys.stderr

    sys.stdout = StreamlitConsoleWriter(old_stdout, placeholder)
    sys.stderr = StreamlitConsoleWriter(old_stderr, placeholder)

    try:
        yield
    except Exception:
        append_console(traceback.format_exc())
        if placeholder is not None:
            render_console(placeholder, height=200)
        raise
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

        if placeholder is not None:
            render_console(placeholder, height=200)

# ============================================================
# UTILIDADES: NORMALIZACIÓN DE NOMBRES
# ============================================================

def normalize_loss_name(loss_name: str) -> str:
    """
    Convierte los nombres mostrados en la GUI al nombre usado por config,
    carpetas, entrenamiento y evaluación.
    """
    if loss_name is None:
        return "l1"

    return LOSS_DISPLAY_TO_KEY.get(
        str(loss_name),
        str(loss_name).lower().replace("-", "_")
    )


def get_group_for_loss(loss_name: str) -> str:
    """
    Devuelve el grupo de la loss: main, experimental o advanced.
    Usa experiment_utils si está disponible y fallback si no.
    """
    loss_name = normalize_loss_name(loss_name)

    if get_loss_group is not None:
        try:
            return get_loss_group(loss_name)
        except Exception:
            pass

    if loss_name in MAIN_LOSSES_FALLBACK:
        return "main"

    if loss_name in EXPERIMENTAL_LOSSES_FALLBACK:
        return "experimental"

    return "advanced"


def safe_filename(name: str) -> str:
    """
    Convierte el nombre de una fuente en un nombre seguro de archivo.
    """
    return (
        str(name)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .lower()
    )


# ============================================================
# UTILIDADES: BÚSQUEDA DE CHECKPOINTS
# ============================================================

def candidate_checkpoint_folders(model_selected: str, loss_name: str, num_sources: int):
    """
    Devuelve posibles carpetas donde puede estar un checkpoint.
    Se incluyen rutas V2/V3 y rutas antiguas para compatibilidad.
    """
    loss_name = normalize_loss_name(loss_name)
    group = get_group_for_loss(loss_name)

    folders = []

    if model_selected == "modelo_de_martin":
        folders.extend([
            CHECKPOINTS_V2_ROOT / group / f"{loss_name} checkpoints" / f"{num_sources}stems",
            CHECKPOINTS_ROOT / "Mis_modelos" / f"{loss_name} checkpoints" / f"{num_sources}stems",
            GUI_DIR / "checkpoints" / "Mis_modelos" / f"{loss_name} checkpoints" / f"{num_sources}stems",
        ])

    elif model_selected == "modelo_de_usuario":
        folders.extend([
            CHECKPOINTS_ROOT / "Modelos_de_usuario" / f"{loss_name} checkpoints" / f"{num_sources}stems",
            GUI_DIR / "checkpoints" / "Modelos_de_usuario" / f"{loss_name} checkpoints" / f"{num_sources}stems",
        ])

    else:
        folders.extend([
            CHECKPOINTS_ROOT / "Modelo_base" / f"{loss_name} checkpoints" / f"{num_sources}stems",
            CHECKPOINTS_ROOT / "Modelo_base",
            GUI_DIR / "checkpoints" / "Modelo_base" / f"{loss_name} checkpoints" / f"{num_sources}stems",
            GUI_DIR / "checkpoints" / "Modelo_base",
        ])

    unique = []
    seen = set()
    for folder in folders:
        folder = folder.resolve()
        if folder not in seen:
            unique.append(folder)
            seen.add(folder)

    return unique


def list_checkpoints(model_selected: str, loss_name: str, num_sources: int):
    """
    Busca checkpoints .pt y .pth recursivamente en las carpetas candidatas.
    """
    folders = candidate_checkpoint_folders(model_selected, loss_name, num_sources)

    candidates = []
    for folder in folders:
        if folder.exists():
            candidates.extend(folder.rglob("*.pt"))
            candidates.extend(folder.rglob("*.pth"))

    candidates = list(dict.fromkeys(candidates))

    candidates = sorted(
        candidates,
        key=lambda p: (
            "best" not in p.name.lower(),
            -p.stat().st_mtime
        )
    )

    return candidates, folders


def checkpoint_label(path: Path) -> str:
    """
    Etiqueta legible para mostrar en el selectbox.
    """
    try:
        rel = path.relative_to(PROJECT_ROOT)
        return str(rel)
    except Exception:
        return str(path)


# ============================================================
# UTILIDADES: CARGA ROBUSTA DE CHECKPOINTS
# ============================================================

def extract_state_dict(checkpoint):
    """
    Soporta varios formatos:
      - Ignite nuevo: checkpoint['model']
      - Otros: checkpoint['state_dict']
      - Otros: checkpoint['model_state_dict']
      - State dict plano.
    """
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in checkpoint:
                return checkpoint[key]

    return checkpoint


def strip_known_prefixes(state_dict):
    """
    Elimina prefijos habituales si aparecen.
    """
    cleaned = OrderedDict()

    for key, value in state_dict.items():
        new_key = key

        for prefix in ("module.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]

        cleaned[new_key] = value

    return cleaned


def infer_model_arch_from_state_dict(state_dict, default_nf: int, default_num_sources: int):
    """
    Intenta inferir la arquitectura recurrente desde el checkpoint.
    Esto evita errores si la config actual no coincide exactamente con el modelo entrenado.
    """
    hidden_size = int(config.config.get("MODEL_HIDDEN_SIZE", 50))
    num_layers = int(config.config.get("MODEL_NUM_LAYERS", 1))
    bidirectional = bool(config.config.get("MODEL_BIDIRECTIONAL", True))
    nf = int(default_nf)

    weight_hh_key = None
    weight_ih_key = None

    for key in state_dict.keys():
        if key.endswith("recurrent_stack.rnn.weight_hh_l0"):
            weight_hh_key = key
        if key.endswith("recurrent_stack.rnn.weight_ih_l0"):
            weight_ih_key = key

    if weight_hh_key is not None:
        hidden_size = int(state_dict[weight_hh_key].shape[1])

    if weight_ih_key is not None:
        nf = int(state_dict[weight_ih_key].shape[1])

    layer_ids = []
    for key in state_dict.keys():
        if "recurrent_stack.rnn.weight_ih_l" in key:
            suffix = key.split("weight_ih_l")[-1]
            layer_str = ""
            for char in suffix:
                if char.isdigit():
                    layer_str += char
                else:
                    break
            if layer_str:
                layer_ids.append(int(layer_str))

    if layer_ids:
        num_layers = max(layer_ids) + 1

    bidirectional = any("_reverse" in key for key in state_dict.keys())

    return {
        "nf": nf,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "bidirectional": bidirectional,
        "num_sources": int(default_num_sources),
    }


def filter_matching_state_dict(model, state_dict):
    """
    Evita que load_state_dict falle por claves existentes con shape incompatible.
    """
    model_state = model.state_dict()
    filtered = OrderedDict()
    skipped = []

    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape == value.shape:
            filtered[key] = value
        else:
            skipped.append(key)

    return filtered, skipped


@st.cache_resource(show_spinner="Cargando modelo...")
def load_separator(checkpoint_path: str, num_sources: int, loss_name: str):
    """
    Carga el separador desde un checkpoint seleccionado.
    """
    device = resolve_device(config.config.get("DEVICE", "auto"))
    checkpoint_path = str(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = extract_state_dict(checkpoint)
    state_dict = strip_known_prefixes(state_dict)

    default_nf = int(config.config["STFT_WINDOW_LENGTH"]) // 2 + 1
    arch = infer_model_arch_from_state_dict(
        state_dict,
        default_nf=default_nf,
        default_num_sources=int(num_sources),
    )

    model = MaskInference.build(
        arch["nf"],
        num_audio_channels=int(config.config["MODEL_NUM_CHANNELS"]),
        hidden_size=int(arch["hidden_size"]),
        num_layers=int(arch["num_layers"]),
        bidirectional=bool(arch["bidirectional"]),
        dropout=float(config.config["MODEL_DROPOUT"]),
        num_sources=int(arch["num_sources"]),
        activation=config.config["MODEL_ACTIVATION"],
    )

    try:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
    except RuntimeError:
        filtered, skipped = filter_matching_state_dict(model, state_dict)
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        print("[WARN] Claves saltadas por shape incompatible:", skipped)

    if missing:
        print("[WARN] Claves faltantes al cargar:", missing)

    if unexpected:
        print("[WARN] Claves inesperadas al cargar:", unexpected)

    model.to(device)
    model.eval()

    return model


# ============================================================
# UTILIDADES: DEPLOY Y ENRUTAMIENTO DE SALIDAS
# ============================================================

def update_runtime_config(model_selected: str, loss_name: str, num_sources: int, checkpoint_path=None):
    """
    Actualiza config para que train/deploy/evaluate usen la selección actual.
    """
    normalized_loss = normalize_loss_name(loss_name)

    config.config["MODEL_NUM_SOURCES"] = int(num_sources)
    config.config["MODEL_LOSS_FUNCTION"] = normalized_loss
    config.config["MODEL_SELECTED_GUI"] = model_selected

    if checkpoint_path is not None:
        config.config["MODEL_PATH"] = str(checkpoint_path)
        config.config["CHECKPOINT_PATH"] = str(checkpoint_path)
        config.config["MODEL_CHECKPOINT_PATH"] = str(checkpoint_path)


def call_deploy(audio_path: str, output_dir: Path, checkpoint_path: Path | None):
    """
    Llama a deploy de forma compatible.
    Si deploy acepta model_path/checkpoint_path, se lo pasa.
    Si no, deja la ruta en config.config.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if checkpoint_path is not None:
        config.config["MODEL_PATH"] = str(checkpoint_path)
        config.config["CHECKPOINT_PATH"] = str(checkpoint_path)
        config.config["MODEL_CHECKPOINT_PATH"] = str(checkpoint_path)

    signature = inspect.signature(deploy)
    kwargs = {
        "output_dir": output_dir,
        "audio_path": audio_path,
    }

    if "model_path" in signature.parameters:
        kwargs["model_path"] = str(checkpoint_path)

    if "checkpoint_path" in signature.parameters:
        kwargs["checkpoint_path"] = str(checkpoint_path)

    return deploy(**kwargs)


def save_audio_signal(audio_signal, output_path: Path):
    """
    Guarda un objeto nussl.AudioSignal como archivo wav.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        audio_signal.write_audio_to_file(str(output_path))
    except Exception:
        if hasattr(audio_signal, "istft"):
            audio_signal.istft()
            audio_signal.write_audio_to_file(str(output_path))
        else:
            raise

    return output_path


def materialize_audio_output(value, source_name: str, output_dir: Path) -> Path:
    """
    Convierte una salida de deploy en ruta de audio.
    Puede recibir:
      - una ruta str/Path;
      - un nussl.AudioSignal;
      - otro objeto compatible con write_audio_to_file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(value, (str, Path)):
        return Path(value)

    if hasattr(value, "write_audio_to_file"):
        output_path = output_dir / f"{safe_filename(source_name)}.wav"
        return save_audio_signal(value, output_path)

    raise TypeError(
        f"No sé convertir la salida de tipo {type(value)} "
        f"para la fuente {source_name}."
    )


def normalize_sources_dict(stem_paths, sources_dict, output_dir: Path):
    """
    Convierte la salida de deploy a un diccionario {source_name: Path}.
    Si deploy devuelve AudioSignal, los guarda como .wav.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(sources_dict, dict) and sources_dict:
        normalized = {}

        for source, value in sources_dict.items():
            normalized[str(source)] = materialize_audio_output(
                value,
                str(source),
                output_dir,
            )

        return normalized

    if isinstance(stem_paths, dict) and stem_paths:
        normalized = {}

        for source, value in stem_paths.items():
            normalized[str(source)] = materialize_audio_output(
                value,
                str(source),
                output_dir,
            )

        return normalized

    if isinstance(stem_paths, (list, tuple)) and stem_paths:
        names_2 = ["vocals", "accompaniment"]
        names_4 = ["vocals", "bass", "drums", "other"]
        names = names_4 if int(config.config["MODEL_NUM_SOURCES"]) == 4 else names_2

        normalized = {}

        for i, value in enumerate(stem_paths):
            source_name = names[i] if i < len(names) else f"source_{i+1}"
            normalized[source_name] = materialize_audio_output(
                value,
                source_name,
                output_dir,
            )

        return normalized

    return {}


# ============================================================
# UTILIDADES: GRÁFICOS
# ============================================================

def make_waveform_plot(audio_path: Path, source_name: str, plot_dir: Path) -> Path:
    """
    Genera y guarda un gráfico de forma de onda.
    """
    plot_dir.mkdir(parents=True, exist_ok=True)

    y, sr = librosa.load(str(audio_path), sr=None, mono=True)

    fig, ax = plt.subplots(figsize=(7, 2.8))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title(f"Waveform: {source_name}")
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Amplitud")
    fig.tight_layout()

    out_path = plot_dir / f"{safe_filename(source_name)}_waveform.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return out_path


def make_spectrogram_plot(audio_path: Path, source_name: str, plot_dir: Path) -> Path:
    """
    Genera y guarda un espectrograma en dB.
    """
    plot_dir.mkdir(parents=True, exist_ok=True)

    y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    stft = librosa.stft(y)
    magnitude = np.abs(stft)

    ref_value = np.max(magnitude)
    if ref_value <= 0:
        ref_value = 1.0

    spectrogram_db = librosa.amplitude_to_db(magnitude, ref=ref_value)

    fig, ax = plt.subplots(figsize=(7, 3.2))
    img = librosa.display.specshow(
        spectrogram_db,
        sr=sr,
        x_axis="time",
        y_axis="hz",
        ax=ax,
    )
    ax.set_title(f"Espectrograma: {source_name}")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()

    out_path = plot_dir / f"{safe_filename(source_name)}_spectrogram.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return out_path


def generate_plots_for_sources(sources_dict, plot_dir: Path):
    """
    Genera waveform y espectrograma para cada stem.
    """
    plot_paths = {}

    for source, audio_path in sources_dict.items():
        audio_path = Path(audio_path)

        if not audio_path.exists():
            continue

        try:
            wave_path = make_waveform_plot(audio_path, source, plot_dir)
            spec_path = make_spectrogram_plot(audio_path, source, plot_dir)

            plot_paths[source] = {
                "waveform": wave_path,
                "spectrogram": spec_path,
            }
        except Exception as exc:
            plot_paths[source] = {
                "error": str(exc)
            }

    return plot_paths


# ============================================================
# UTILIDADES: RENDER DE RESULTADOS
# ============================================================

def render_audio_results():
    """
    Renderiza audios separados y botones de descarga.
    """
    sources_dict = st.session_state.get("last_sources_dict", {})

    if not sources_dict:
        st.info("Todavía no hay resultados de separación.")
        return

    st.subheader("Stems separados")

    for source, path in sources_dict.items():
        path = Path(path)

        st.markdown(f"**{source}**")

        if not path.exists():
            st.warning(f"No se encontró el archivo de audio: `{path}`")
            continue

        st.audio(str(path))

        with open(path, "rb") as file:
            st.download_button(
                label=f"Descargar {source}",
                data=file,
                file_name=f"{safe_filename(source)}.wav",
                mime="audio/wav",
                key=f"download_audio_{source}",
            )


def render_plot_results():
    """
    Renderiza waveforms y espectrogramas generados.
    """
    plot_paths = st.session_state.get("last_plot_paths", {})

    st.subheader("Gráficos de salida")

    if not plot_paths:
        st.info("Los gráficos aparecerán aquí después de separar un audio.")
        return

    for source, paths in plot_paths.items():
        st.markdown(f"**{source}**")

        if "error" in paths:
            st.warning(f"No se pudieron generar gráficos para {source}: {paths['error']}")
            continue

        waveform = Path(paths["waveform"])
        spectrogram = Path(paths["spectrogram"])

        if waveform.exists():
            st.image(str(waveform), caption=f"Waveform - {source}", use_container_width=True)

        if spectrogram.exists():
            st.image(str(spectrogram), caption=f"Espectrograma - {source}", use_container_width=True)


def render_loaded_model_status():
    """
    Muestra estado actual del modelo cargado.
    """
    loaded_model = st.session_state.get("loaded_model_name", "Ninguno")
    loaded_path = st.session_state.get("loaded_model_path", None)
    loaded_loss = st.session_state.get("loaded_loss_name", None)
    loaded_sources = st.session_state.get("loaded_num_sources", None)

    st.subheader(f"Modelo cargado actualmente: **{loaded_model}**")

    if loaded_path:
        st.caption(f"Loss: `{loaded_loss}` | Stems: `{loaded_sources}`")
        st.caption(f"Checkpoint: `{loaded_path}`")
    else:
        st.caption("No se ha cargado ningún checkpoint todavía.")


# ============================================================
# ESTADO DE SESIÓN
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "main"

for key in [
    "loaded_model_name",
    "loaded_model_path",
    "loaded_loss_name",
    "loaded_num_sources",
    "last_sources_dict",
    "last_stem_paths",
    "last_plot_paths",
    "last_run_dir",
]:
    if key not in st.session_state:
        if key in ["last_sources_dict", "last_stem_paths", "last_plot_paths"]:
            st.session_state[key] = {}
        else:
            st.session_state[key] = None


# ============================================================
# SIDEBAR: SELECCIÓN DE CONFIGURACIÓN
# ============================================================

st.sidebar.title("Funcionalidades para tu propio modelo")

num_sources_selection = st.sidebar.selectbox(
    "Número de fuentes (stems)",
    options=[2, 4],
    index=0,
)

model_selected = st.sidebar.selectbox(
    "Modelos disponibles",
    options=["self", "modelo_de_martin", "modelo_de_usuario"],
    index=1,
)

with st.sidebar.expander("Sobre el modelo", expanded=True):
    desc = help_quotes.model_descriptions.get(
        model_selected,
        "Sin descripción disponible para este modelo."
    )
    st.markdown(desc)

if model_selected == "modelo_de_martin":
    database_selection = st.sidebar.selectbox(
        "Bases de datos disponibles",
        options=["MUSDB18", "Rock DB", "HipHop DB"],
        index=0,
        disabled=True,
        key="db_locked_martin",
    )
else:
    database_selection = st.sidebar.selectbox(
        "Bases de datos disponibles",
        options=["MUSDB18", "Rock DB", "HipHop DB"],
    )

default_loss_index = 1 if model_selected == "modelo_de_martin" else 0
loss_fn_selection = st.sidebar.selectbox(
    "Función de pérdida",
    options=LOSS_OPTIONS,
    index=default_loss_index,
)

normalized_loss_name = normalize_loss_name(loss_fn_selection)
normalized_num_sources = int(num_sources_selection)

with st.sidebar.expander("Sobre la función de pérdida", expanded=True):
    if loss_fn_selection in loss_functions_help.loss_key_map:
        loss_functions_help.loss_key_map[loss_fn_selection]()
    elif normalized_loss_name in loss_functions_help.loss_key_map:
        loss_functions_help.loss_key_map[normalized_loss_name]()
    else:
        st.markdown("Sin descripción disponible para esta función de pérdida.")

with st.sidebar.expander("Sobre la base de datos", expanded=True):
    desc = help_quotes.database_descriptions.get(
        database_selection,
        "Sin descripción disponible para esta base de datos."
    )
    st.markdown(desc)


# ============================================================
# SIDEBAR: CHECKPOINTS DISPONIBLES
# ============================================================

available_checkpoints, searched_folders = list_checkpoints(
    model_selected=model_selected,
    loss_name=normalized_loss_name,
    num_sources=normalized_num_sources,
)

selected_checkpoint = None

st.sidebar.markdown("---")
st.sidebar.subheader("Checkpoint")

if available_checkpoints:
    selected_checkpoint = st.sidebar.selectbox(
        "Checkpoint disponible",
        available_checkpoints,
        format_func=checkpoint_label,
    )
else:
    st.sidebar.warning("No hay checkpoints para esta configuración.")
    with st.sidebar.expander("Carpetas buscadas"):
        for folder in searched_folders:
            st.code(str(folder))


# ============================================================
# SIDEBAR: BOTONES DE ACCIÓN
# ============================================================

if st.sidebar.button("Cargar configuración seleccionada", use_container_width=True):
    update_runtime_config(
        model_selected=model_selected,
        loss_name=normalized_loss_name,
        num_sources=normalized_num_sources,
        checkpoint_path=selected_checkpoint,
    )
    init_gui_console()

    st.session_state.loaded_model_name = model_selected
    st.session_state.loaded_loss_name = normalized_loss_name
    st.session_state.loaded_num_sources = normalized_num_sources

    if selected_checkpoint is None:
        st.session_state.loaded_model_path = None
        st.warning("No se ha encontrado ningún checkpoint para la configuración seleccionada.")
        with st.expander("Carpetas buscadas"):
            for folder in searched_folders:
                st.code(str(folder))
    else:
        st.session_state.loaded_model_path = str(selected_checkpoint)
        st.success("Configuración cargada correctamente.")
        st.caption(f"Checkpoint seleccionado: `{selected_checkpoint}`")

if st.sidebar.button("Entrenar modelo", use_container_width=True):
    
    config.config["CHECKPOINTS_GROUP"] = "Modelos_de_usuario"
    checkpoint_dir(normalized_loss_name,normalized_num_sources,"Modelos_de_usuario")
    config.config["TRAIN_VERSION"] = "gui"
    
    st.session_state.page = "train"
    st.rerun()

if st.sidebar.button("Configuración avanzada", use_container_width=True):
    st.session_state.page = "advanced_config"
    st.rerun()

if st.sidebar.button("Ayuda de parámetros", use_container_width=True):
    st.session_state.page = "param_help"
    st.rerun()

# ============================================================
# PÁGINA: AYUDA DE PARÁMETROS
# ============================================================

if st.session_state.page == "param_help":
    st.title("Ayuda de parámetros")

    st.markdown(
        "En esta sección se muestra una descripción resumida de los parámetros "
        "configurables desde la interfaz avanzada."
    )

    st.text_area(
        "Descripción de parámetros",
        value=get_parameter_help_text(),
        height=600,
    )

    if st.button("Volver a configuración avanzada", use_container_width=True):
        st.session_state.page = "advanced_config"
        st.rerun()

    st.stop()

# ============================================================
# PÁGINA: ENTRENAMIENTO
# ============================================================

if st.session_state.page == "train":
    st.title("Entrenamiento del modelo")

    st.info(
        "Esta sección lanza el entrenamiento con la configuración actual. "
        "El proceso puede tardar bastante dependiendo de la pérdida y del número de stems."
    )

    if st.button("Entrenar", use_container_width=True):
        update_runtime_config(
            model_selected=model_selected,
            loss_name=normalized_loss_name,
            num_sources=normalized_num_sources,
            checkpoint_path=selected_checkpoint,
        )

        train_console_placeholder = st.empty()
        render_console(train_console_placeholder, height=200)

        with st.spinner("Entrenando modelo..."):
            with capture_logs_to_gui(train_console_placeholder, "Iniciando entrenamiento"):
                logs = run_training_and_capture_logs()

        append_console(logs)
        render_console(train_console_placeholder, height=200)

        st.success("Entrenamiento completado.")
        st.subheader("Registro de entrenamiento:")
        st.text_area("Logs", logs, height=400)

    if st.session_state.get("loaded_model_path"):
        if st.checkbox("¿Quieres descargar el modelo cargado actualmente?"):
            model_path = Path(st.session_state.loaded_model_path)

            if model_path.exists():
                temp_path = GUI_RESULTS_ROOT / "temp_model.pth"
                shutil.copy(model_path, temp_path)

                with open(temp_path, "rb") as file:
                    st.download_button(
                        label="Descargar modelo (.pth/.pt)",
                        data=file,
                        file_name=model_path.name,
                        mime="application/octet-stream",
                    )
            else:
                st.warning("El checkpoint cargado ya no existe en disco.")

    if st.button("Volver", use_container_width=True):
        st.session_state.page = "main"
        st.rerun()

    st.stop()


# ============================================================
# PÁGINA: CONFIGURACIÓN AVANZADA
# ============================================================

if st.session_state.page == "advanced_config":
    st.title("Configuración avanzada para tu modelo")
    st.markdown("Modifica los parámetros de entrenamiento y del modelo.")

    if st.button("Ver ayuda de parámetros", use_container_width=True):
        st.session_state.page = "param_help"
        st.rerun()


    col1_advanced, col2_advanced = st.columns([2, 2])

    with col1_advanced:
        stft_param_selection()
        train_param_selection()
        mix_gen_param_selection()

    with col2_advanced:
        model_param_selection()
        fx_param_selection()

    col_save, col_cancel = st.columns(2)

    with col_save:
        if st.button("Guardar y volver", use_container_width=True):
            save_and_exit_param_selection()

    with col_cancel:
        if st.button("Cancelar", use_container_width=True):
            st.session_state.page = "main"
            st.rerun()

    st.stop()


# ============================================================
# PÁGINA PRINCIPAL
# ============================================================

st.title("Separación de música por stems")

render_loaded_model_status()

# ============================================================
# CONSOLA
# ============================================================
col_console = st.container()

with col_console:
    st.subheader("Consola y logs")

    console_placeholder = st.empty()
    render_console(console_placeholder, height=200)

    if st.button("Limpiar consola", use_container_width=False):
        clear_console()
        render_console(console_placeholder, height=200)

st.markdown("---")

col_audio, col_graphs = st.columns([1, 1])

# ============================================================
# COLUMNA 1: AUDIO Y RESULTADOS
# ============================================================

with col_audio:
    st.subheader("Audio de entrada")

    uploaded_file = st.file_uploader(
        "Sube un archivo de audio",
        type=["wav", "mp3"],
        accept_multiple_files=False,
    )

    if uploaded_file:
        st.audio(uploaded_file, format="audio/wav")

        if st.button("Separar audio", use_container_width=True):
            checkpoint_path = st.session_state.get("loaded_model_path", None)

            if checkpoint_path is None:
                st.error("Primero carga una configuración con checkpoint desde la barra lateral.")
                st.stop()

            update_runtime_config(
                model_selected=st.session_state.get("loaded_model_name", model_selected),
                loss_name=st.session_state.get("loaded_loss_name", normalized_loss_name),
                num_sources=int(st.session_state.get("loaded_num_sources", normalized_num_sources)),
                checkpoint_path=checkpoint_path,
            )

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=Path(uploaded_file.name).suffix
            ) as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_file_path = tmp_file.name

            run_id = time.strftime("%Y%m%d_%H%M%S")
            run_dir = GUI_RESULTS_ROOT / f"run_{run_id}"
            audio_out_dir = run_dir / "audio"
            plot_out_dir = run_dir / "plots"

            audio_out_dir.mkdir(parents=True, exist_ok=True)
            plot_out_dir.mkdir(parents=True, exist_ok=True)

            st.info("Iniciando separación de audio...")
            progress_bar = st.progress(0, text="Preparando...")

            with st.spinner("Separando audio con el modelo..."):
                for percent in [10, 25, 40, 60, 80]:
                    time.sleep(0.15)
                    progress_bar.progress(percent, text=f"Procesando... {percent}%")

                with capture_logs_to_gui(console_placeholder, "Iniciando separación de audio"):
                    deploy_result = call_deploy(
                        audio_path=tmp_file_path,
                        output_dir=audio_out_dir,
                        checkpoint_path=Path(checkpoint_path),
                    )

                if isinstance(deploy_result, tuple) and len(deploy_result) == 2:
                    stem_paths, sources_dict = deploy_result
                else:
                    stem_paths, sources_dict = deploy_result, {}

                sources_dict = normalize_sources_dict(stem_paths, sources_dict,audio_out_dir)

                progress_bar.progress(90, text="Generando gráficos...")
                with capture_logs_to_gui(console_placeholder, "Generando gráficos de salida"):
                    plot_paths = generate_plots_for_sources(sources_dict, plot_out_dir)

            progress_bar.progress(100, text="Completado.")

            st.session_state.last_sources_dict = {
                source: str(path)
                for source, path in sources_dict.items()
            }
            st.session_state.last_stem_paths = {
                source: str(path)
                for source, path in sources_dict.items()
            }
            st.session_state.last_plot_paths = {
                source: {
                    key: str(value)
                    for key, value in paths.items()
                }
                for source, paths in plot_paths.items()
            }
            st.session_state.last_run_dir = str(run_dir)

            st.success("Separación completada.")
            st.caption(f"Resultados guardados en: `{run_dir}`")

            append_console("Separación completada correctamente.")
            render_console(console_placeholder, height=200)

        render_audio_results()

        if st.button("Evaluar modelo cargado", use_container_width=True):
            checkpoint_path = st.session_state.get("loaded_model_path", None)

            if checkpoint_path is None:
                st.error("Primero carga una configuración con checkpoint desde la barra lateral.")
                st.stop()

            with capture_logs_to_gui(console_placeholder, "Iniciando evaluación del modelo"):
                separator = load_separator(
                    checkpoint_path,
                    int(st.session_state.get("loaded_num_sources", normalized_num_sources)),
                    st.session_state.get("loaded_loss_name", normalized_loss_name),
                )

                evaluation(frames=5, separator=separator)

            append_console("Evaluación finalizada correctamente.")
            render_console(console_placeholder, height=200)

            st.success("Evaluación finalizada.")

    else:
        st.info("Sube un archivo para interactuar con el modelo.")





# ============================================================
# COLUMNA 3: GRÁFICOS
# ============================================================

with col_graphs:
    render_plot_results()

    if st.session_state.get("last_run_dir"):
        st.caption(f"Última carpeta de resultados: `{st.session_state.last_run_dir}`")
