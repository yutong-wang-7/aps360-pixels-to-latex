"""metrics.py: metrics of a trained model (accuracy, top-5, macro-F1, pairs merged).

PIPELINE  data_collection/: 1 get_fonts -> 2 make_dataset -> 3 data_figures -> 4 hasy
          training/:        5 tune (optional) -> 6 train
          evaluation/:      7 final_test (once, at the end)
THIS FILE helper: predictions and metrics
READS     datasets + info from data.py
WRITES    nothing (returns dicts)
USED BY   training/tune.py, training/train.py, evaluation/final_test.py, evaluation/equation_eval.py
RUN       not run directly
------------------------------------------------------------------------------

Every metric is computed twice:
  normal          the prediction must be exactly the true class
  pairs merged    a prediction anywhere in the true class's near-identical group
                  (symbols.NEAR_IDENTICAL, e.g. o / O / 0) counts as correct
"""
import json

import numpy as np
import torch

from data import get_loader
from models import build_model
from symbols import NEAR_IDENTICAL
from train_loop import device


def load_trained(name, K, results, suffix=""):
    """Rebuild a model with the size saved in results/<name>.json and load its weights
    (results/<name><suffix>.pt, e.g. suffix "_ft" for the fine-tuned model)."""
    r = json.loads((results / f"{name}.json").read_text(encoding="utf-8"))
    net, _ = build_model(name, K, r.get("model_config"))
    net.load_state_dict(torch.load(results / f"{name}{suffix}.pt", map_location=device))
    return net.to(device), r


def predict(net, dataset, batch_size=1024):
    """Logits (CPU tensor, N x K) for every image of a dataset, in order."""
    net.to(device).eval()
    out = []
    with torch.no_grad():
        for inputs, _ in get_loader(dataset, batch_size, shuffle=False):
            out.append(net(inputs.to(device)).cpu())
    return torch.cat(out)


def macro_f1(y, p, K):
    """Unweighted mean of the per-class F1 over the classes present in y."""
    f1 = []
    for k in range(K):
        tp = np.sum((p == k) & (y == k)); fp = np.sum((p == k) & (y != k)); fn = np.sum((p != k) & (y == k))
        if tp + fn == 0:          # class absent from this split (coverage gap)
            continue
        f1.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f1))


def group_map(names):
    """G[class id] = id of the first class in its near-identical group (itself if none)."""
    G = np.arange(len(names))
    for grp in NEAR_IDENTICAL:
        ids = [names.index(s) for s in grp if s in names]
        if ids:
            G[ids] = ids[0]
    return G


def metrics(logits, y, K, G):
    """Top-1, top-5, macro-F1, and top-1 / macro-F1 with near-identical groups merged."""
    p = logits.argmax(1).numpy()
    top5 = logits.topk(5, 1).indices.numpy()
    return dict(n=int(len(y)), acc=float((p == y).mean()),
                top5=float((top5 == y[:, None]).any(1).mean()), macro_f1=macro_f1(y, p, K),
                acc_merged=float((G[p] == G[y]).mean()), macro_f1_merged=macro_f1(G[y], G[p], K))


def evaluate_printed(net, dataset, info, K, G, fonts):
    """Metrics of one printed split: clean images, augmented images, and clean accuracy
    per font. Also returns the predicted class of every image (for figures)."""
    logits = predict(net, dataset)
    y, clean, aug = info["y"], info["aug"] == 0, info["aug"] == 1
    t = torch.from_numpy
    out = dict(clean=metrics(logits[t(clean)], y[clean], K, G),
               aug=metrics(logits[t(aug)], y[aug], K, G))
    out["per_font_clean_acc"] = {}
    for f in np.unique(info["font"]):
        m = clean & (info["font"] == f)
        out["per_font_clean_acc"][fonts[f]["name"]] = metrics(logits[t(m)], y[m], K, G)["acc"]
    return out, logits.argmax(1).numpy()


def evaluate_hasy(net, dataset, info, K, G):
    """Metrics on all images of one HASY (handwriting) split, plus the predictions."""
    logits = predict(net, dataset)
    return metrics(logits, info["y"], K, G), logits.argmax(1).numpy()
