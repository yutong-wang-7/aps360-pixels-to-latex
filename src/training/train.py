r"""train.py: step 6, train the final ANN and CNN, evaluate on validation data, write results.

PIPELINE  data_collection/: 1 get_fonts -> 2 make_dataset -> 3 data_figures -> 4 hasy
          training/:        5 tune (optional) -> 6 train
          evaluation/:      7 final_test (once, at the end)
THIS FILE step 6; the driver. The pieces live in
            training/data.py        load datasets (printed splits, HASYv2 writer split)
            training/models.py      ANN baseline, CNN primary
            training/train_loop.py  train_net / evaluate / get_model_name (Lab 3 template)
            evaluation/metrics.py   accuracy, top-5, macro-F1, pairs merged
            evaluation/plotting.py  learning curves, error / handwriting figures
            evaluation/report.py    LaTeX tables and macros
READS     data/{train,val}.npz + meta.json, data/hasy.npz (optional),
          results/tuning/best_<model>.json (from tune.py, optional)
WRITES    results/<model>.pt|.json (+ <model>_ft.pt), results/ckpt/, results/curves/,
          figures/ (tables, macros, figures)
USED BY   evaluation/final_test.py, the report (Baseline / Primary Model sections)
RUN       python src/training/train.py [--load] [--finetune] [--models cnn]
------------------------------------------------------------------------------

Per model:
  1. size + learning rate from tune.py if it has been run (else the defaults in
     models.py and --lr)
  2. train (train_net) on all training images; keep the epoch with the lowest error on
     the CLEAN validation images (aug == 0)
  3. evaluate on printed val (clean + augmented)
  4. test zero-shot on handwriting (HASY val writers), if data/hasy.npz exists
  5. with --finetune: continue training on printed + HASY train images (class-balanced
     sampling, 16 epochs), select on HASY val writers, and check printed val accuracy again
  6. error analysis: most frequent handwriting confusions on the HASY val writers
Test data is not touched here: that is evaluation/final_test.py.
"""
import argparse, json, os, shutil, sys
from pathlib import Path

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(SRC, d) for d in ("data_collection", "training", "evaluation")]

import numpy as np
import torch
from torch.utils.data import ConcatDataset

from data import balanced_weights, load_data
from metrics import evaluate_hasy, evaluate_printed, group_map
from models import MODELS, build_model, count_params
from plotting import fig_curves, fig_errors, fig_hasy
from report import (confusions_table, hasy_confusions, hasy_table, near_identical_tex, results_macros,
                    results_table)
from train_loop import device, load_curves, set_seed, train_net


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--results", default="results")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--models", nargs="+", default=["ann", "cnn"], choices=list(MODELS))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3, help="used when there is no tuning result")
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=360)
    ap.add_argument("--no-tuned", action="store_true", help="ignore results/tuning/best_<model>.json")
    ap.add_argument("--load", action="store_true", help="load results/<model>.pt instead of training")
    ap.add_argument("--finetune", action="store_true", help="fine-tune on printed + HASY train images")
    ap.add_argument("--ft-epochs", type=int, default=16)   # best epoch is selected on HASY val
    ap.add_argument("--ft-lr", type=float, default=3e-4)
    return ap.parse_args()


def tuned_config(name, a):
    """(model_config, learning_rate, batch_size, source) from tune.py's result, or the defaults."""
    p = a.results / "tuning" / f"best_{name}.json"
    if a.no_tuned or not p.exists():
        return {}, a.lr, a.bs, "defaults"
    t = json.loads(p.read_text())
    return t["model_config"], t["learning_rate"], t.get("batch_size", a.bs), str(p)


def train_or_load(name, d, a):
    """Lab 3 flow: build the model, train_net, load the best checkpoint.
    With --load, reuse results/<name>.pt + .json from an earlier run instead."""
    pt, js = a.results / f"{name}.pt", a.results / f"{name}.json"
    if a.load and pt.exists() and js.exists():
        old = json.loads(js.read_text(encoding="utf-8"))
        try:
            net, cfg = build_model(name, d["K"], old.get("model_config"))
            net.load_state_dict(torch.load(pt, map_location=device))
            print(f"\n{name}: loaded {pt} ({cfg})")
            keep = ("params", "best_epoch", "train_time_s", "history", "model_config", "learning_rate")
            return net.to(device), {k: old[k] for k in keep}
        except (RuntimeError, KeyError, TypeError):  # saved by an older train.py
            print(f"\n{name}: {pt} is from an older train.py, training instead")

    model_config, lr, bs, source = tuned_config(name, a)
    set_seed(a.seed)
    net, cfg = build_model(name, d["K"], model_config)
    print(f"\n{name}: {cfg}, lr {lr}, batch size {bs} (from {source}); {count_params(net):,} parameters, "
          f"{len(d['train'][0]):,} training images, device {device}")
    best_path, _, curve_path, elapsed = train_net(
        net, d["train"][0], d["val_clean"], batch_size=bs, learning_rate=lr,
        num_epochs=a.epochs, weight_decay=a.wd, results=str(a.results))
    net.load_state_dict(torch.load(best_path, map_location=device))   # best epoch
    shutil.copy(best_path, pt)
    hist = load_curves(curve_path)
    return net, dict(params=count_params(net), best_epoch=int(np.argmin(hist["val_err"])) + 1,
                     train_time_s=round(elapsed, 1), history=hist, model_config=cfg,
                     learning_rate=lr, batch_size=bs)


def finetune(net, name, d, a):
    """Continue training `net` (lower learning rate) on printed + handwritten images;
    select the epoch with the lowest error on the HASY val writers. Batches are drawn
    class-balanced: half printed, half handwritten, every class equally likely in each
    half (HASY's training images are dominated by a few symbols such as infinity)."""
    train_set = ConcatDataset([d["train"][0], d["hasy"]["train"][0]])
    weights = balanced_weights(d["train"][1]["y"], d["hasy"]["train"][1]["y"])
    set_seed(a.seed)
    print(f"\n{name} fine-tuning: {len(train_set):,} training images")
    best_path, _, curve_path, elapsed = train_net(
        net, train_set, d["hasy"]["val"][0], batch_size=a.bs, learning_rate=a.ft_lr,
        num_epochs=a.ft_epochs, weight_decay=a.wd, tag="_ft", results=str(a.results),
        sample_weights=weights)
    net.load_state_dict(torch.load(best_path, map_location=device))
    shutil.copy(best_path, a.results / f"{name}_ft.pt")
    hist = load_curves(curve_path)
    return net, dict(best_epoch=int(np.argmin(hist["val_err"])) + 1,
                     train_time_s=round(elapsed, 1), history=hist,
                     config=dict(epochs=a.ft_epochs, lr=a.ft_lr, data="printed train + HASY train",
                                 sampling="class-balanced, half printed / half handwritten",
                                 selection="HASY val writers"))


def main():
    a = get_args()
    figs = Path(a.figures); a.results = Path(a.results)
    a.results.mkdir(parents=True, exist_ok=True); figs.mkdir(parents=True, exist_ok=True)
    device_name = (f"an {torch.cuda.get_device_name(0)} GPU" if device.type == "cuda" else "a laptop CPU")
    print("device:", device, device_name)

    # data (train + val only; the test split is for final_test.py)
    d = load_data(a.data, a.seed, splits=("train", "val"))
    K, names, fonts, hasy = d["K"], d["names"], d["fonts"], d["hasy"]
    val_set, val_info = d["val"]
    G = group_map(names) # near-identical groups
    print(f"train {len(d['train'][0])}, val {len(val_set)} ({len(d['val_clean'])} clean), K = {K}")
    if hasy is not None:
        print("HASY:", {s: len(hasy[s][0]) for s in hasy}, "writers val/test:",
              len(np.unique(hasy["val"][1]["user"])), len(np.unique(hasy["test"][1]["user"])))
    elif a.finetune:
        sys.exit("--finetune needs data/hasy.npz (run src/data_collection/hasy.py)")

    # per model: train or load -> printed val metrics -> handwriting zero-shot -> fine-tune
    runs, preds, hpreds = {}, {}, {}
    for name in a.models:
        net, r = train_or_load(name, d, a)
        r["config"] = dict(epochs=a.epochs, bs=a.bs, wd=a.wd, seed=a.seed,
                           optimizer="Adam", schedule="cosine", loss="cross-entropy")
        r["val"], preds[name] = evaluate_printed(net, val_set, val_info, K, G, fonts)
        v = r["val"]["clean"]
        print(f"{name}: best epoch {r['best_epoch']}, val clean acc {v['acc']:.3f}, top-5 {v['top5']:.3f}, "
              f"macro-F1 {v['macro_f1']:.3f}, merged acc {v['acc_merged']:.3f}, "
              f"val aug acc {r['val']['aug']['acc']:.3f}")
        print("  per-font clean val acc:", {k: round(x, 3) for k, x in r["val"]["per_font_clean_acc"].items()})

        if hasy is not None:
            r["hasy_zero"] = {}
            r["hasy_zero"]["val"], hpreds[name.upper()] = evaluate_hasy(net, *hasy["val"], K, G)
            z = r["hasy_zero"]["val"]
            print(f"  HASY zero-shot (val writers): acc {z['acc']:.3f}, merged {z['acc_merged']:.3f}, "
                  f"top-5 {z['top5']:.3f}")

        if a.finetune:
            net, ft = finetune(net, name, d, a)
            ft["hasy_val"], hpreds[name.upper() + " ft"] = evaluate_hasy(net, *hasy["val"], K, G)
            ft["printed_val"], _ = evaluate_printed(net, val_set, val_info, K, G, fonts)
            r["hasy_ft"] = ft
            print(f"  fine-tuned: HASY val acc {ft['hasy_val']['acc']:.3f} "
                  f"(merged {ft['hasy_val']['acc_merged']:.3f}), printed val acc "
                  f"{ft['printed_val']['clean']['acc']:.3f}")
        elif not a.load:
            # a newly trained model makes an older fine-tuned checkpoint stale: remove it
            (a.results / f"{name}_ft.pt").unlink(missing_ok=True)

        (a.results / f"{name}.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
        runs[name] = r

    # ---- report outputs (tables, LaTeX macros, figures)
    fig_curves(runs, figs)
    results_table(runs, figs, False)
    results_macros(runs, figs, False, device_name,
                   {s: hasy[s][1] for s in hasy} if hasy is not None else None)
    near_identical_tex(figs)
    if hasy is not None:
        hasy_table(runs, figs)
        fig_hasy(hasy["val"][1], hpreds, names, figs)
        # error analysis: the best handwriting model available (fine-tuned CNN if present)
        label = next((k for k in ("CNN ft", "CNN", "ANN ft", "ANN") if k in hpreds), None)
        if label is not None:
            c = hasy_confusions(hasy["val"][1], hpreds[label], names, G, figs, label)
            (a.results / "hasy_confusions.json").write_text(json.dumps(c, indent=1), encoding="utf-8")
            print(f"\n{label} handwriting errors (val writers): {c['errors']}, of which "
                  f"{c['errors_in_groups']} ({c['errors_in_groups'] / max(c['errors'], 1):.0%}) inside "
                  f"near-identical groups; top {len(c['top'])} pairs = {c['top_k_share']:.0%} of errors")
            for row in c["top"]:
                print(f"  {row['true']:>14} -> {row['pred']:<14} {row['count']:3d}/{row['of']:<4d}"
                      + ("  (merged group)" if row["merged"] else ""))
    if "cnn" in preds:
        fig_errors(val_info, preds, names, fonts, figs)
        confusions_table(val_info, preds, names, figs)
    print(f"\nwrote {a.results}/ and {figs}/")


if __name__ == "__main__":
    main()
