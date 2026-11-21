r"""tune.py: step 5 (optional), grid search over GRID for the ANN and the CNN (as in Lab 3).

READS   data/  (train, clean val, HASY val)
WRITES  results/tuning/best_<model>.json  best setting per model (train.py uses it)
        results/tuning/grid.json          every setting with its curve
        figures/tuning_table.tex, figures/fig_tuning.{pdf,png}
RUN     python src/training/tune.py [--models cnn] [--epochs 6] [--train-frac 0.3]

Each setting trains for --epochs on a random --train-frac of the training images.
Score = mean of printed val error (best epoch) and zero-shot HASY handwriting val error
(same checkpoint). Test data is never used here.
"""
import argparse, itertools, json, os, sys
from pathlib import Path

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(SRC, d) for d in ("data_collection", "training", "evaluation")]

import numpy as np
import torch
import torch.nn as nn

from data import get_loader, load_data, random_subset
from models import build_model, count_params
from plotting import fig_tuning
from report import tuning_table
from train_loop import device, evaluate, load_curves, set_seed, train_net

# everything that is searched: training settings + model sizes (every combination is run)
LEARNING_RATES = [3e-4, 1e-3, 3e-3, 1e-2]
BATCH_SIZES = [64, 256]
GRID = {
    # 4 x 2 x 2 = 16 settings
    "ann": dict(learning_rate=LEARNING_RATES, batch_size=BATCH_SIZES, hidden1=[128, 256]),
    # 4 x 2 x 3 x 2 x 2 = 96 settings
    "cnn": dict(learning_rate=LEARNING_RATES, batch_size=BATCH_SIZES, num_layers=[2, 3, 4],
                channels=[16, 32], hidden_units=[128, 256]),
}
TRAINING_KEYS = ("learning_rate", "batch_size")   # the rest of a setting goes to build_model
WEIGHT_DECAY, SEED = 1e-4, 360
OUT, FIGS = Path("results/tuning"), Path("figures")


def split(config):
    """Split a setting into (training settings, model sizes)."""
    train = {k: v for k, v in config.items() if k in TRAINING_KEYS}
    sizes = {k: v for k, v in config.items() if k not in TRAINING_KEYS}
    return train, sizes


def run(name, config, d, train_set, epochs):
    """Train one setting; return its record for grid.json."""
    train, sizes = split(config)

    # build the model with this setting's sizes
    set_seed(SEED)
    net, _ = build_model(name, d["K"], sizes)
    print(f"\n{name} {config}: {count_params(net):,} parameters")

    # train on printed images, keeping the epoch with the lowest printed val error
    best_path, val_err, curve_path, _ = train_net(
        net, train_set, d["val_clean"], batch_size=train["batch_size"],
        learning_rate=train["learning_rate"], num_epochs=epochs, weight_decay=WEIGHT_DECAY,
        tag="_" + "_".join(f"{k}{v}" for k, v in sizes.items()), results=str(OUT))
    hist = load_curves(curve_path)

    # handwriting error of that best epoch on HASY val (zero-shot)
    net.load_state_dict(torch.load(best_path, map_location=device))
    hasy_loader = get_loader(d["hasy"]["val"][0], 256, shuffle=False)
    hasy_err, _ = evaluate(net, hasy_loader, nn.CrossEntropyLoss())

    # score = mean of the printed and handwriting errors
    score = (val_err + hasy_err) / 2
    print(f"  printed val err {val_err:.4f}, handwriting val err {hasy_err:.4f}, score {score:.4f}")

    return dict(config=config, params=count_params(net), best_epoch=int(np.argmin(hist["val_err"])) + 1,
                best_val_err=float(val_err), hasy_val_err=float(hasy_err), score=float(score),
                history=hist)


def main():
    # command-line options
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(GRID), choices=list(GRID))
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--train-frac", type=float, default=0.3)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    # data: a random fraction of the training images keeps each run short
    d = load_data("data", SEED)
    train_set = random_subset(d["train"][0], a.train_frac, SEED)

    # start from earlier results so models not tuned this time are kept
    grid_path = OUT / "grid.json"
    grid = json.loads(grid_path.read_text()) if grid_path.exists() else {}

    for name in a.models:
        # run every combination of GRID[name]
        keys = list(GRID[name])
        configs = [dict(zip(keys, v)) for v in itertools.product(*GRID[name].values())]
        grid[name] = [run(name, c, d, train_set, a.epochs) for c in configs]

        # save the lowest-score setting for train.py
        best = min(grid[name], key=lambda r: r["score"])["config"]
        train, sizes = split(best)
        print(f"\nbest {name}: {best}")
        (OUT / f"best_{name}.json").write_text(json.dumps(dict(**train, model_config=sizes), indent=1))

    # save all settings and make the report table + figure
    grid_path.write_text(json.dumps(grid, indent=1))
    tuning_table(grid, FIGS)
    fig_tuning(grid, FIGS)


if __name__ == "__main__":
    main()
