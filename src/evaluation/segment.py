r"""segment.py: split an equation image into symbols (classical, no learning).

PIPELINE  data_collection/: 1 get_fonts -> 2 make_dataset -> 2b make_equations -> 3 data_figures -> 4 hasy
          training/:        5 tune (optional) -> 6 train
          evaluation/:      7 final_test (once, at the end); equation_eval (stage 2)
THIS FILE helper + a check of the segmenter against the ground-truth boxes (no PyTorch needed)
READS     data/equations/<split>.jsonl + images (from make_equations.py)
WRITES    figures/segment_macros.tex, results/segment_<split>.json (when run directly)
USED BY   evaluation/equation_eval.py
RUN       python src/evaluation/segment.py [--split val]
------------------------------------------------------------------------------

Proposal, stage 2: "segmentation uses connected components, merging multi-part symbols
(i, =, \div) by overlap and ordering left to right".

  1. ink = pixels darker than INK (same threshold as the ground-truth boxes)
  2. 8-connected components (label propagation + pointer jumping, numpy only);
     components smaller than MIN_AREA pixels are noise
  3. merge components that overlap horizontally by at least MERGE_OVERLAP of the
     narrower one (the dot of i, the bars of =, the parts of \leq, \div, :, ;, !)
  4. order the merged segments left to right
Known failure: neighbours that touch (tight spacing, italic overhang) become one segment.
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np
from PIL import Image

INK = 96             # ink threshold (of 255); make_equations.py uses the same for its boxes
MIN_AREA = 3         # smaller components are noise
MERGE_OVERLAP = 0.5  # merge if x-overlap >= this fraction of the narrower component


def load_ink(path):
    """Equation PNG (black on white) -> uint8 array with ink = 255 on 0."""
    return 255 - np.asarray(Image.open(path).convert("L"))


def label_components(ink):
    """8-connected component labels of a boolean image (background = -1)."""
    H, W = ink.shape
    BG = H * W
    lab = np.where(ink, np.arange(H * W).reshape(H, W), BG)
    flat_ink = ink.ravel()
    while True:
        p = np.pad(lab, 1, constant_values=BG)
        m = lab
        for dy in (0, 1, 2):                                    # min over the 3x3 neighbourhood
            for dx in (0, 1, 2):
                m = np.minimum(m, p[dy:dy + H, dx:dx + W])
        f = np.where(flat_ink, m.ravel(), BG)
        for _ in range(8):                                      # pointer jumping: follow labels
            f = np.where(flat_ink, np.append(f, BG)[f], BG)
        new = f.reshape(H, W)
        if np.array_equal(new, lab):
            return np.where(ink, new, -1)
        lab = new


def segment(img):
    """img: uint8 ink = 255. Returns (segments, labels): each segment is a dict with the
    box (x0, y0, x1, y1) and the component labels it contains, sorted left to right."""
    ink = img > INK
    lab = label_components(ink)
    comps = []
    for l in np.unique(lab[lab >= 0]):
        ys, xs = np.where(lab == l)
        if len(xs) >= MIN_AREA:
            comps.append(dict(box=[int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                              labels=[int(l)]))
    merged = True
    while merged:                                               # merge until nothing changes
        merged = False
        comps.sort(key=lambda c: c["box"][0])
        for i in range(len(comps)):
            for j in range(i + 1, len(comps)):
                a, b = comps[i]["box"], comps[j]["box"]
                if b[0] >= a[2]:
                    break                                       # sorted by x0: no later overlap
                ov = min(a[2], b[2]) - max(a[0], b[0])
                if ov >= MERGE_OVERLAP * min(a[2] - a[0], b[2] - b[0]):
                    comps[i] = dict(box=[min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])],
                                    labels=comps[i]["labels"] + comps[j]["labels"])
                    del comps[j]
                    merged = True
                    break
            if merged:
                break
    comps.sort(key=lambda c: (c["box"][0] + c["box"][2]) / 2)
    return comps, lab


def crop(img, lab, seg):
    """The pixels of one segment only (neighbours that intrude into its box are removed)."""
    x0, y0, x1, y1 = seg["box"]
    keep = np.isin(lab[y0:y1, x0:x1], seg["labels"])
    return np.where(keep, img[y0:y1, x0:x1], 0).astype(np.uint8)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def match(pred_boxes, gt_boxes, thr=0.5):
    """Greedy one-to-one matching by IoU >= thr. Returns {gt index: pred index}."""
    pairs = sorted(((iou(p, g), i, j) for j, p in enumerate(pred_boxes) for i, g in enumerate(gt_boxes)),
                   reverse=True)
    used_p, used_g, m = set(), set(), {}
    for v, i, j in pairs:
        if v < thr:
            break
        if i not in used_g and j not in used_p:
            m[i] = j; used_g.add(i); used_p.add(j)
    return m


def check(data, split, limit=None):
    """Segment every equation of a split and compare with the ground-truth boxes."""
    root = Path(data) / "equations"
    rows = [json.loads(l) for l in (root / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if l]
    rows = rows[:limit] if limit else rows
    n_ok = n_sym = n_found = 0
    fails = {}
    for r in rows:
        segs, _ = segment(load_ink(root / r["file"]))
        m = match([s["box"] for s in segs], r["boxes"])
        n_sym += len(r["boxes"]); n_found += len(m)
        ok = len(segs) == len(r["boxes"]) and len(m) == len(r["boxes"])
        n_ok += ok
        for i, t in enumerate(r["tokens"]):
            if i not in m:
                fails[t] = fails.get(t, 0) + 1
    return dict(split=split, equations=len(rows), equations_ok=n_ok / len(rows),
                symbols=n_sym, symbol_recall=n_found / n_sym,
                most_missed=sorted(fails.items(), key=lambda kv: -kv[1])[:10])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--results", default="results")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--split", default="val")
    a = ap.parse_args()
    res = check(a.data, a.split)
    Path(a.results).mkdir(parents=True, exist_ok=True); Path(a.figures).mkdir(parents=True, exist_ok=True)
    (Path(a.results) / f"segment_{a.split}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    S = a.split.capitalize()
    (Path(a.figures) / "segment_macros.tex").write_text(
        f"\\newcommand{{\\Seg{S}EqOk}}{{{100 * res['equations_ok']:.1f}}}\n"
        f"\\newcommand{{\\Seg{S}SymRecall}}{{{100 * res['symbol_recall']:.1f}}}\n", encoding="utf-8")
    print(f"{a.split}: {res['equations']} equations, {100 * res['equations_ok']:.1f}% segmented exactly; "
          f"{100 * res['symbol_recall']:.1f}% of {res['symbols']} symbols found (IoU >= 0.5)")
    print("most missed symbols:", res["most_missed"])


if __name__ == "__main__":
    main()
