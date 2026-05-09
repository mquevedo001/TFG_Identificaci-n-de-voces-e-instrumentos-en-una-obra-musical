import torch
from pathlib import Path

paths = [
    Path(r"checkpoints\Mis_modelos\log_compressed_l2 checkpoints\2stems\best_checkpoint_69_val_loss=-99.5427.pt"),
]

for p in paths:
    print("\n===", p, "===")
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    for k, v in sd.items():
        if any(x in k for x in [
            "rnn.weight_ih_l0",
            "rnn.weight_hh_l0",
            "embedding",
            "batch_norm",
        ]):
            print(k, tuple(v.shape))