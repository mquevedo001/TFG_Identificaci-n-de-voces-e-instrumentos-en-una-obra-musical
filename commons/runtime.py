import torch


def resolve_device(requested="auto"):
    if requested in (None, "auto"):
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    if requested == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA no disponible. Se utilizará CPU.")
        return torch.device("cpu")

    return torch.device(requested)