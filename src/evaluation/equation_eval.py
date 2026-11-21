r"""equation_eval.py: stage 2, read whole equations: segment -> classify each symbol.

PIPELINE  data_collection/: 1 get_fonts -> 2 make_dataset -> 2b make_equations -> 3 data_figures -> 4 hasy
          training/:        5 tune (optional) -> 6 train
          evaluation/:      7 final_test (once, at the end); equation_eval (stage 2)
THIS FILE stage-2 evaluation of the trained single-symbol models (nothing is trained here)
READS     data/equations/<split>.jsonl + images (make_equations.py), data/meta.json,
          results/<model>.pt|.json (train.py)
WRITES    results/equations_<split>.json, figures/equations_table.tex,
          figures/equations_eval_macros.tex, figures/fig_equation_preds.{pdf,png}
USED BY   the report
RUN       python src/evaluation/equation_eval.py [--models ann cnn] [--ft] [--test]
------------------------------------------------------------------------------

Proposal: "for stage 2, I will report equation-level exact match, with segmentation and
classification errors counted separately". Per model:
  oracle symbol acc   classify the ground-truth symbol boxes (classification errors only)
  segmentation ok     equations whose segments match the ground truth one-to-one (IoU >= 0.5)
  pipeline symbol acc symbols found by the segmenter and classified correctly / all symbols
  exact match         whole equation right: same number of segments, every label right
  exact (merged)      the same with near-identical groups merged (o / O / 0, ...)
Crops are normalized exactly like the single-symbol data (crop to ink, pad to a square,
area-resize so the longest side is 28 px, centre in 32x32), so relative size is lost,
as for single symbols. Validation fonts by default; --test only for the final report.
"""
import argparse, json, os, sys
from pathlib import Path

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(SRC, d) for d in ("data_collection", "training", "evaluation")]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import TensorDataset

from data import load_meta, to_tensor
from metrics import group_map, load_trained, predict
from models import MODELS
from plotting import NAME, save
from render import place, to_square
from segment import crop, load_ink, match, segment


def normalize(arr):
    """Crop of ink (uint8, ink = 255) -> 32x32 like the single-symbol data."""
    im = Image.fromarray(arr)
    return np.asarray(place(to_square(im)), np.uint8) if im.getbbox() else np.zeros((32, 32), np.uint8)


def prepare(data, split):
    """Segment every equation once; collect normalized crops for the ground-truth boxes
    (oracle) and for the segmenter's output (pipeline)."""
    root = Path(data) / "equations"
    rows = [json.loads(l) for l in (root / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if l]
    oracle, pipe = [], []
    for r in rows:
        img = load_ink(root / r["file"])
        for b in r["boxes"]:
            oracle.append(normalize(img[b[1]:b[3], b[0]:b[2]]))
        segs, lab = segment(img)
        r["segs"] = [s["box"] for s in segs]
        r["match"] = match(r["segs"], r["boxes"])              # gt index -> segment index
        for s in segs:
            pipe.append(normalize(crop(img, lab, s)))
    return rows, np.stack(oracle), np.stack(pipe)


def score(rows, p_oracle, p_pipe, G):
    """All stage-2 metrics from the predicted labels of the oracle and pipeline crops."""
    io = ip = 0
    o_ok = o_n = seg_ok = sym_ok = exact = exact_m = 0
    for r in rows:
        y = np.array(r["labels"]); n = len(y)
        po = p_oracle[io:io + n]; io += n
        pp = p_pipe[ip:ip + len(r["segs"])]; ip += len(r["segs"])
        o_ok += int((po == y).sum()); o_n += n
        good_seg = len(r["segs"]) == n and len(r["match"]) == n
        seg_ok += good_seg
        sym_ok += sum(int(pp[int(j)] == y[int(i)]) for i, j in r["match"].items())
        if good_seg:
            seq = pp[[r["match"][i] for i in range(n)]]
            exact += bool((seq == y).all()); exact_m += bool((G[seq] == G[y]).all())
        r["pred"] = pp.tolist()
    E = len(rows)
    return dict(equations=E, symbols=o_n, oracle_symbol_acc=o_ok / o_n, segmentation_ok=seg_ok / E,
                pipeline_symbol_acc=sym_ok / o_n, exact_match=exact / E, exact_match_merged=exact_m / E)


def fig_preds(rows, names, data, label, out, n=5):
    """A few equations with the model's reading (green: whole equation right)."""
    root = Path(data) / "equations"
    pick = rows[:n]
    fig, axes = plt.subplots(len(pick), 1, figsize=(6.4, 0.75 * len(pick)))
    for ax, r in zip(np.atleast_1d(axes), pick):
        ax.imshow(255 - load_ink(root / r["file"]), cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([]); ax.set_yticks([])
        pred = " ".join(names[k] for k in r["pred"])
        ok = pred == " ".join(r["tokens"])
        ax.set_title(f"{label}: {pred}", fontsize=5.5, loc="left", pad=1, family="monospace",
                     color="#2e7d32" if ok else "#8b1e3f")
    fig.tight_layout(pad=0.2, h_pad=0.4)
    save(fig, out, "fig_equation_preds")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--results", default="results")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--models", nargs="+", default=["ann", "cnn"], choices=list(MODELS))
    ap.add_argument("--ft", action="store_true", help="use the fine-tuned weights (<model>_ft.pt)")
    ap.add_argument("--test", action="store_true", help="test-font equations (final report only)")
    a = ap.parse_args()
    split = "test" if a.test else "val"
    results, figs = Path(a.results), Path(a.figures)
    figs.mkdir(parents=True, exist_ok=True)

    meta = load_meta(a.data)
    names = [c["latex"] for c in meta["classes"]]; K = len(names)
    mean, std = meta["splits"]["train"]["pixel_mean"], meta["splits"]["train"]["pixel_std"]
    G = group_map(names)
    rows, X_oracle, X_pipe = prepare(a.data, split)
    ds = lambda X: TensorDataset(to_tensor(X, mean, std), torch.zeros(len(X), dtype=torch.long))
    print(f"{split}: {len(rows)} equations, {len(X_oracle)} symbols, {len(X_pipe)} segments")

    res, last = {}, None
    for name in a.models:
        net, _ = load_trained(name, K, results, suffix="_ft" if a.ft else "")
        p_or = predict(net, ds(X_oracle)).argmax(1).numpy()
        p_pi = predict(net, ds(X_pipe)).argmax(1).numpy()
        res[name] = score(rows, p_or, p_pi, G)
        r = res[name]
        print(f"{name}: oracle symbol acc {r['oracle_symbol_acc']:.3f} | segmentation ok "
              f"{r['segmentation_ok']:.3f} | pipeline symbol acc {r['pipeline_symbol_acc']:.3f} | "
              f"exact match {r['exact_match']:.3f} (merged {r['exact_match_merged']:.3f})")
        last = name

    (results / f"equations_{split}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    pc = lambda x: f"{100 * x:.1f}"
    lines = [f"{NAME[m]} & {pc(r['oracle_symbol_acc'])} & {pc(r['pipeline_symbol_acc'])} & "
             f"{pc(r['exact_match'])} & {pc(r['exact_match_merged'])} \\\\" for m, r in res.items()]
    seg = next(iter(res.values()))["segmentation_ok"]
    (figs / "equations_table.tex").write_text(
        "\\begin{tabular}{lrrrr}\n\\toprule\n"
        "Model & Symbol acc (true boxes) & Symbol acc (segmented) & Exact match & Exact (merged) \\\\\n"
        "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
    mac = [f"\\newcommand{{\\EqSegOk}}{{{pc(seg)}}}"]
    for m, r in res.items():
        P = m.capitalize()
        mac += [f"\\newcommand{{\\{P}EqOracleAcc}}{{{pc(r['oracle_symbol_acc'])}}}",
                f"\\newcommand{{\\{P}EqPipeAcc}}{{{pc(r['pipeline_symbol_acc'])}}}",
                f"\\newcommand{{\\{P}EqExact}}{{{pc(r['exact_match'])}}}",
                f"\\newcommand{{\\{P}EqExactMerged}}{{{pc(r['exact_match_merged'])}}}"]
    (figs / "equations_eval_macros.tex").write_text("\n".join(mac) + "\n", encoding="utf-8")
    fig_preds(rows, names, a.data, last.upper(), figs)
    print(f"wrote {results}/equations_{split}.json, {figs}/equations_table.tex, "
          f"equations_eval_macros.tex, fig_equation_preds")


if __name__ == "__main__":
    main()
