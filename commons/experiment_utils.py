from pathlib import Path
import json
from config import config

MAIN_LOSSES = [
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
]

EXPERIMENTAL_LOSSES = [
    "deep_feature",
    "deep_feature_emd",
]

ADVANCED_LOSSES = [
    "lmrs",
    "l_mrs",
    "lpsa_phase",
]



def get_loss_group(loss_name: str) -> str:
    loss_name = str(loss_name).lower()

    if loss_name in EXPERIMENTAL_LOSSES:
        return "experimental"

    if loss_name in ADVANCED_LOSSES:
        return "advanced"

    return "main"


def get_checkpoint_base() -> Path:
    root = Path(config.config.get("CHECKPOINTS_ROOT", "checkpoints"))
    group = config.config.get("CHECKPOINTS_GROUP", "Mis_modelos_v2")
    return root / group


def get_results_base() -> Path:
    root = Path(config.config.get("RESULTS_ROOT", "resultados_modelos"))
    group = config.config.get("RESULTS_EVAL_GROUP", "evaluate_v2")
    return root / group


def checkpoint_dir(loss_name: str, num_sources: int) -> Path:
    group_root = config.config.get("CHECKPOINTS_GROUP", "Mis_modelos_v2")
    loss_group = get_loss_group(loss_name)

    return (
        Path(config.config["CHECKPOINTS_ROOT"])
        / group_root
        / loss_group
        / f"{loss_name} checkpoints"
        / f"{num_sources}stems"
    )


def has_checkpoint(path: Path) -> bool:
    return path.exists() and (any(path.glob("*.pth")) or any(path.glob("*.pt")))
def legacy_checkpoint_dir(loss_name: str, num_sources: int) -> Path:
    return (Path(config.config["CHECKPOINTS_ROOT"]) / "Mis modelos" / f"{loss_name} checkpoints" / f"{int(num_sources)}stems")



def resolve_checkpoint_dir(loss_name: str, num_sources: int) -> Path:
    v2 = checkpoint_dir(loss_name, num_sources)
    legacy = legacy_checkpoint_dir(loss_name, num_sources)

    if has_checkpoint(v2):
        return v2

    if has_checkpoint(legacy):
        return legacy

    raise FileNotFoundError(
        f"No se encontró ningún checkpoint para "
        f"loss={loss_name}, sources={num_sources}. "
        f"Buscado en '{v2}' y '{legacy}'."
    )


def eval_results_dir(loss_name: str, num_sources: int) -> Path:
    return (get_results_base()/ get_loss_group(loss_name)/ loss_name/ f"{int(num_sources)}stems" )

def training_config_snapshot(loss_name: str, num_sources: int) -> dict:
    return {
        "version": config.config.get("TRAIN_VERSION", "v2"),
        "loss": str(loss_name),
        "group": get_loss_group(loss_name),
        "num_sources": int(num_sources),
        "model": {
            "hidden_size": int(config.config["MODEL_HIDDEN_SIZE"]),
            "num_layers": int(config.config["MODEL_NUM_LAYERS"]),
            "bidirectional": bool(config.config["MODEL_BIDIRECTIONAL"]),
            "dropout": float(config.config["MODEL_DROPOUT"]),
            "activation": config.config["MODEL_ACTIVATION"],
            "num_channels": int(config.config["MODEL_NUM_CHANNELS"]),
        },
        "stft": {
            "window_length": int(config.config["STFT_WINDOW_LENGTH"]),
            "hop_length": int(config.config["STFT_HOP_LENGTH"]),
            "window_type": config.config["STFT_WINDOW_TYPE"],
        },
        "training": {
            "max_epochs": int(config.config["MAX_EPOCHS"]),
            "epoch_length": int(config.config["EPOCH_LENGTH"]),
            "learning_rate": float(config.config["LEARNING_RATE"]),
            "weight_decay": float(config.config.get("WEIGHT_DECAY", 0.0)),
            "gradient_clip": float(config.config.get("GRADIENT_CLIP", 1.0)),
            "early_stopping_patience": int(config.config.get("EARLY_STOPPING_PATIENCE", 15)),
        },
    }


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)