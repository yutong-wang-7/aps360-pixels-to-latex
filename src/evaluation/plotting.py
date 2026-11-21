"""plotting.py: figures of training curves and predictions.

PIPELINE  data_collection/: 1 get_fonts -> 2 make_dataset -> 3 data_figures -> 4 hasy
          training/:        5 tune (optional) -> 6 train
          evaluation/:      7 final_test (once, at the end)
THIS FILE helper: matplotlib figures
READS     curves / predictions passed in by train.py
WRITES    figures/fig_curves, fig_errors, fig_hasy, fig_tuning (.pdf for LaTeX, .png for viewing)
USED BY   training/train.py, training/tune.py
RUN       not run directly
------------------------------------------------------------------------------

  plot_training_curve   one run's error and loss curves (Lab 3 template); handy for
                        quick checks, e.g. plot_training_curve(r["curve_path"])
  fig_curves            report figure: learning curves of all models in one plot
  fig_errors            report figure: CNN mistakes on clean validation images
  fig_hasy              report figure: handwritten images with each model's prediction
  fig_tuning            report figure: validation accuracy per epoch of every tuning run
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "savefig.dpi": 300})
COLOR = {"ann": "#7a5c00", "cnn": "#1f4e79"}
NAME = {"ann": "ANN (baseline)", "cnn": "CNN (primary)"}
GREEN, RED = "#2e7d32", "#8b1e3f"


def save(fig, out, name):
    """Save a figure as PDF (for LaTeX) and PNG (for viewing), then close it."""
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_training_curve(path, out=None):
    """ Plots the training curve for a model run, given the csv files
    containing the train/validation error/loss.

    Args:
        path: The base path of the csv files produced during training
        out: file name to save the figure to (default: show it on screen)
    """
    train_err = np.loadtxt("{}_train_err.csv".format(path))
    val_err = np.loadtxt("{}_val_err.csv".format(path))
    train_loss = np.loadtxt("{}_train_loss.csv".format(path))
    val_loss = np.loadtxt("{}_val_loss.csv".format(path))
    n = len(train_err) # number of epochs
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3))
    a1.set_title("Train vs Validation Error")
    a1.plot(range(1, n + 1), train_err, label="Train")
    a1.plot(range(1, n + 1), val_err, label="Validation")
    a1.set_xlabel("Epoch"); a1.set_ylabel("Error"); a1.legend(loc='best')
    a2.set_title("Train vs Validation Loss")
    a2.plot(range(1, n + 1), train_loss, label="Train")
    a2.plot(range(1, n + 1), val_loss, label="Validation")
    a2.set_xlabel("Epoch"); a2.set_ylabel("Loss"); a2.legend(loc='best')
    fig.tight_layout()
    if out:
        fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    else:
        plt.show()


def fig_curves(runs, out):
    """Learning curves of all models: loss and accuracy per epoch (train dashed, val solid)."""
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.1))
    for name, r in runs.items():
        h, c = r["history"], COLOR[name]
        ep = np.arange(1, len(h["train_err"]) + 1)
        ax[0].plot(ep, h["train_loss"], c=c, ls="--", lw=1)
        ax[0].plot(ep, h["val_loss"], c=c, lw=1.4, label=NAME[name])
        ax[1].plot(ep, (1 - np.array(h["train_err"])) * 100, c=c, ls="--", lw=1)
        ax[1].plot(ep, (1 - np.array(h["val_err"])) * 100, c=c, lw=1.4, label=NAME[name])
    ax[0].set_ylabel("cross-entropy loss"); ax[1].set_ylabel("accuracy (%)")
    for x in ax:
        x.set_xlabel("epoch"); x.spines[["top", "right"]].set_visible(False)
    ax[0].plot([], [], c="gray", ls="--", lw=1, label="train (augmented)")
    ax[0].plot([], [], c="gray", lw=1.4, label="val (clean, unseen fonts)")
    ax[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, out, "fig_curves")


def fig_errors(info, preds, names, fonts, out, n=12):
    """CNN mistakes on clean validation images (with the ANN's guess)."""
    y, clean = info["y"], info["aug"] == 0
    wrong = np.where(clean & (preds["cnn"] != y))[0]
    rng = np.random.default_rng(0)
    pick = rng.choice(wrong, min(n, len(wrong)), replace=False) if len(wrong) else []
    cols = 6; rows = max(1, int(np.ceil(len(pick) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 0.8, rows * 1.5))
    for ax in np.atleast_1d(axes).flat:
        ax.axis("off")
    for ax, i in zip(np.atleast_1d(axes).flat, pick):
        ax.imshow(255 - info["raw"][i], cmap="gray", vmin=0, vmax=255)
        txt = f"{fonts[info['font'][i]]['name']}\ntrue {names[y[i]]}\nCNN {names[preds['cnn'][i]]}"
        if "ann" in preds:
            txt += f"\nANN {names[preds['ann'][i]]}"
        ax.set_title(txt, fontsize=6.5, linespacing=1.05)
    fig.tight_layout(pad=0.2, h_pad=0.4)
    save(fig, out, "fig_errors")


def fig_hasy(info, preds, names, out, n=18):
    """Random handwritten val images with each model's prediction (green right, red wrong)."""
    y = info["y"]
    pick = np.random.default_rng(0).choice(len(y), min(n, len(y)), replace=False)
    cols = 9; rows = int(np.ceil(len(pick) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 0.75, rows * 1.25))
    for ax in np.atleast_1d(axes).flat:
        ax.axis("off")
    for ax, i in zip(np.atleast_1d(axes).flat, pick):
        ax.imshow(255 - info["raw"][i], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"true {names[y[i]]}", fontsize=6, pad=2)
        for k, (lab, p) in enumerate(preds.items()):
            ok = p[i] == y[i]
            ax.text(0.5, -0.08 - 0.16 * k, f"{lab} {names[p[i]]}", transform=ax.transAxes,
                    ha="center", va="top", fontsize=5.5, color=GREEN if ok else RED)
    fig.subplots_adjust(wspace=0.15, hspace=0.9)
    save(fig, out, "fig_hasy")


# the tuning table and figure show only the best TOP_K settings per model
TOP_K = 5
SHORT = dict(learning_rate="lr", batch_size="bs", num_layers="layers", channels="ch",
             hidden_units="hidden", hidden1="hidden")


def top_runs(runs, k=TOP_K):
    """The k tuning runs with the lowest score (mean val error), best first."""
    return sorted(runs, key=lambda r: r.get("score", r["best_val_err"]))[:k]


def short_config(config):
    """Compact label for a tuning setting, e.g. 'lr 0.001, bs 64, layers 3'."""
    return ", ".join(f"{SHORT.get(k, k)} {v:g}" for k, v in config.items())


def fig_tuning(grid, out):
    """Printed validation accuracy per epoch for the best TOP_K settings of the search
    (one panel per model; the selected setting, lowest mean val error, is drawn thick)."""
    models = list(grid)
    fig, axes = plt.subplots(1, len(models), figsize=(3.3 * len(models), 2.2), squeeze=False)
    for ax, name in zip(axes[0], models):
        # one curve per top setting, best (first) drawn thick
        for i, r in enumerate(top_runs(grid[name])):
            acc = (1 - np.array(r["history"]["val_err"])) * 100
            ax.plot(np.arange(1, len(acc) + 1), acc, lw=2.0 if i == 0 else 0.9,
                    alpha=1 if i == 0 else 0.7, label=short_config(r["config"]))
        ax.set_title(f"{NAME[name]}, best {min(TOP_K, len(grid[name]))} of {len(grid[name])}",
                     fontsize=8)
        ax.set_xlabel("epoch")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=5)
    axes[0][0].set_ylabel("val accuracy (%)")
    fig.tight_layout()
    save(fig, out, "fig_tuning")
