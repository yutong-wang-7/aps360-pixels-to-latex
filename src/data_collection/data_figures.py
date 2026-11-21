"""data_figures.py: step 3, figures/tables about the dataset for the report.

PIPELINE  data_collection/: 1 get_fonts -> 2 make_dataset -> 3 data_figures -> 4 hasy
          training/:        5 tune (optional) -> 6 train
          evaluation/:      7 final_test (once, at the end)
THIS FILE step 3
READS     data/ (step 2)
WRITES    figures/fig_*.{pdf,png}, figures/dataset_table.tex, figures/qa/
USED BY   the report (Data Processing section)
RUN       python src/data_collection/data_figures.py
------------------------------------------------------------------------------

Figures and tables for the Data Processing section, plus QA sheets.

  python src/data_collection/data_figures.py --data data --out figures

Report figures (PDF + PNG):
  fig_fonts_grid      clean renders of selected symbols in every font, grouped by split
  fig_augmented       clean render + augmented training samples for a few classes
  fig_coverage        fraction of each symbol group each font covers
  fig_class_counts    training images per class (sorted)
  dataset_table.tex   LaTeX table: fonts / classes / images per split
QA only (check every glyph by eye, e.g. phi vs varphi in each font):
  qa/clean_<font>.png all classes, clean render, labelled
"""
import argparse, json, os, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from symbols import CLASSES, GROUPS

SPLIT_COLOR = {"train": "#1f4e79", "val": "#7a5c00", "test": "#8b1e3f"}
SHOW = ["a", "g", "0", "\\epsilon", "\\varepsilon", "\\phi", "\\varphi", "\\partial",
        "\\Sigma", "\\xi", "\\leq", "\\notin", "\\Rightarrow", "\\{", "\\infty", "\\nabla"]
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "savefig.dpi": 300})


def save(fig, out, name):
    """Save a figure as both PDF (for LaTeX) and PNG (for quick viewing), then close it."""
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def label(latex):
    return latex  # plain LaTeX command text; avoids mathtext gaps for some commands


def fig_fonts_grid(clean, ok, meta, out):
    """Grid of clean renders: rows = fonts (coloured by split), columns = the SHOW symbols."""
    fonts = meta["fonts"]
    order = sorted(range(len(fonts)), key=lambda i: ["train", "val", "test"].index(fonts[i]["split"]))
    idx = [meta["original_classes"].index(s) for s in SHOW]
    fig, axes = plt.subplots(len(order), len(idx), figsize=(len(idx) * 0.42, len(order) * 0.42 + 0.3))
    axes = np.atleast_2d(axes)
    for r, fi in enumerate(order):
        for c, ci in enumerate(idx):
            ax = axes[r, c]
            ax.imshow(255 - clean[fi, ci] if ok[fi, ci] else np.full((32, 32), 235),
                      cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(label(SHOW[c]), fontsize=5.5, rotation=60, ha="left", va="bottom")
        f = fonts[fi]
        axes[r, 0].set_ylabel(f"{f['name']} ({f['split']})", rotation=0, ha="right", va="center",
                              fontsize=6.5, color=SPLIT_COLOR[f["split"]])
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    save(fig, out, "fig_fonts_grid")


def fig_augmented(data, meta, out, n=9):
    """For a few classes: the clean render (boxed) next to n augmented training images."""
    d = np.load(data / "train.npz")
    names = [c["latex"] for c in meta["classes"]]
    pick = [names.index(s) for s in ["\\varepsilon", "\\phi", "x", "\\leq", "\\Sigma"] if s in names]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(len(pick), n + 1, figsize=((n + 1) * 0.45, len(pick) * 0.48))
    for r, k in enumerate(pick):
        clean = np.where((d["y"] == k) & (d["aug"] == 0))[0]
        augi = rng.choice(np.where((d["y"] == k) & (d["aug"] == 1))[0], n, replace=False)
        for c, i in enumerate([clean[0]] + list(augi)):
            ax = axes[r, c]
            ax.imshow(255 - d["X"][i], cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(c == 0); s.set_color("#1f4e79"); s.set_linewidth(1.2)
        axes[r, 0].set_ylabel(label(names[k]), rotation=0, ha="right", va="center", fontsize=7)
    axes[0, 0].set_title("clean", fontsize=7)
    axes[0, (n + 1) // 2].set_title("augmented (rotation, scale, shift, stroke, blur, noise)", fontsize=7)
    fig.subplots_adjust(wspace=0.05, hspace=0.08)
    save(fig, out, "fig_augmented")


def fig_coverage(ok, meta, out):
    """Heatmap: fraction of each symbol group that each font can render."""
    fonts = meta["fonts"]
    G = np.zeros((len(GROUPS), len(fonts)))
    for gi, g in enumerate(GROUPS):
        cls = [i for i, c in enumerate(CLASSES) if c["group"] == g]
        G[gi] = ok[:, cls].mean(1)
    fig, ax = plt.subplots(figsize=(0.32 * len(fonts) + 1.6, 2.0))
    im = ax.imshow(G, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(fonts)))
    ax.set_xticklabels([f["name"] for f in fonts], rotation=60, ha="right")
    for t, f in zip(ax.get_xticklabels(), fonts):
        t.set_color(SPLIT_COLOR[f["split"]])
    ax.set_yticks(range(len(GROUPS))); ax.set_yticklabels([g.replace("_", " ") for g in GROUPS])
    for gi in range(len(GROUPS)):
        for fi in range(len(fonts)):
            if G[gi, fi] < 1:
                ax.text(fi, gi, f"{G[gi, fi]:.0%}", ha="center", va="center", fontsize=5,
                        color="white" if G[gi, fi] > 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.03, label="classes covered")
    save(fig, out, "fig_coverage")


def fig_class_counts(data, meta, out):
    """Bar chart of training images per class, sorted (shows the class balance)."""
    y = np.load(data / "train.npz")["y"]
    K = len(meta["classes"])
    cnt = np.bincount(y, minlength=K)
    o = np.argsort(cnt)
    fig, ax = plt.subplots(figsize=(6.0, 1.6))
    ax.bar(range(K), cnt[o], width=1.0, color="#1f4e79", edgecolor="white", linewidth=0.3)
    ax.set_xlim(-0.5, K - 0.5); ax.set_xticks([])
    ax.set_xlabel(f"{K} classes (sorted by count)"); ax.set_ylabel("training images")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.5); ax.set_axisbelow(True)
    ax.set_title(f"min {cnt.min()}, median {int(np.median(cnt))}, max {cnt.max()}", fontsize=7)
    save(fig, out, "fig_class_counts")


def dataset_table(meta, out):
    """LaTeX table: fonts and clean / augmented / total images per split."""
    rows = []
    for s in ["train", "val", "test"]:
        st = meta["splits"][s]
        rows.append(f"{s.capitalize()} & {len(st['fonts'])} & {', '.join(st['fonts'])} & "
                    f"{st['clean']:,} & {st['augmented']:,} & {st['images']:,} \\\\")
    K = len(meta["classes"])
    tex = ("\\begin{table}[h]\n\\centering\\small\n"
           f"\\caption{{Stage-1 dataset ({K} classes, split by font, seed {meta['seed']}).}}\n"
           "\\label{tab:data}\n\\begin{tabular}{lrp{5.2cm}rrr}\n\\toprule\n"
           "Split & Fonts & Font names & Clean & Augmented & Total \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    (out / "dataset_table.tex").write_text(tex, encoding="utf-8")


def qa_sheets(clean, ok, meta, out):
    """One sheet per font with every class's clean render, for checking glyphs by eye
    (missing classes in red). Not for the report."""
    q = out / "qa"; q.mkdir(exist_ok=True)
    C = len(CLASSES); cols = 16; rows = int(np.ceil(C / cols))
    for fi, f in enumerate(meta["fonts"]):
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 0.7, rows * 0.8))
        for i, ax in enumerate(axes.flat):
            ax.set_xticks([]); ax.set_yticks([])
            if i >= C:
                ax.axis("off"); continue
            ax.imshow(255 - clean[fi, i] if ok[fi, i] else np.full((32, 32), 200),
                      cmap="gray", vmin=0, vmax=255)
            ax.set_title(CLASSES[i]["latex"], fontsize=6, color="black" if ok[fi, i] else "red")
        fig.suptitle(f"{f['name']} ({f['split']}): {int(ok[fi].sum())}/{C} classes", fontsize=9)
        fig.subplots_adjust(wspace=0.1, hspace=0.6)
        fig.savefig(q / f"clean_{f['name']}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)


def main():
    """Load data/meta.json and data/clean.npz, then write every figure and table."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((data / "meta.json").read_text(encoding="utf-8"))
    cz = np.load(data / "clean.npz"); clean, ok = cz["clean"], cz["ok"]
    fig_fonts_grid(clean, ok, meta, out)
    fig_augmented(data, meta, out)
    fig_coverage(ok, meta, out)
    fig_class_counts(data, meta, out)
    dataset_table(meta, out)
    qa_sheets(clean, ok, meta, out)
    print(f"figures written to {out}/")


if __name__ == "__main__":
    main()
