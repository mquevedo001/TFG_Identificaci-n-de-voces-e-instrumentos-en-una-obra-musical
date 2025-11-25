import os
import re
import pandas as pd
import matplotlib.pyplot as plt

def parse_logs(logs_folder="logs_train_runs"):
    # Regex para extraer epoch y val_loss
    pattern = re.compile(r"\[Validator\] Epoch (\d+) - val_loss: ([0-9\.eE+-]+)")
    records = []

    for file in os.listdir(logs_folder):
        if file.endswith(".out"):
            filepath = os.path.join(logs_folder, file)
            with open(filepath, "r") as f:
                text = f.read()

            # Extraer lossfn y num_sources del nombre del archivo
            match_name = re.match(r"train_(.+)_(\d+)src\.out", file)
            if not match_name:
                continue
            lossfn, num_sources = match_name.groups()

            for line in text.splitlines():
                m = pattern.search(line)
                if m:
                    epoch = int(m.group(1))
                    val_loss = float(m.group(2))
                    records.append({
                        "lossfn": lossfn,
                        "num_sources": int(num_sources),
                        "epoch": epoch,
                        "val_loss": val_loss
                    })

    return pd.DataFrame(records)


def summarize_results(df):
    if df.empty:
        print(" No se encontraron registros en los logs.")
        return pd.DataFrame()

    summary = (
        df.groupby(["lossfn", "num_sources"])
        .apply(lambda g: pd.Series({
            "best_val_loss": g["val_loss"].min(),
            "best_epoch": g.loc[g["val_loss"].idxmin(), "epoch"],
            "last_epoch": g["epoch"].max()
        }))
        .reset_index()
        .sort_values("best_val_loss")
    )

    return summary


def plot_val_loss(df):
    for (lossfn, num_sources), group in df.groupby(["lossfn", "num_sources"]):
        plt.figure(figsize=(8, 5))
        plt.plot(group["epoch"], group["val_loss"], marker="o")
        plt.title(f"Loss={lossfn}, Sources={num_sources}")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Loss")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    logs_folder = "logs_train_runs"
    df = parse_logs(logs_folder)

    if not df.empty:
        print("\n Mejor resultado por modelo:\n")
        summary = summarize_results(df)
        print(summary)

        print("\nRanking global (ordenado por mejor val_loss):\n")
        print(summary[["lossfn", "num_sources", "best_val_loss", "best_epoch"]])

        # Dibujar curvas de validación
        plot_val_loss(df)
