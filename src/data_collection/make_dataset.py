"""make_dataset.py: step 2, render every symbol in every font -> train/val/test images.

PIPELINE  data_collection/: 1 get_fonts -> 2 make_dataset -> 3 data_figures -> 4 hasy
          training/:        5 tune (optional) -> 6 train
          evaluation/:      7 final_test (once, at the end)
THIS FILE step 2 (~1-3 min on CPU, parallel over fonts)
READS     fonts/ (step 1), symbols.CLASSES, fonts_config.FONTS
WRITES    data/{train,val,test}.npz, clean.npz, meta.json, class_counts.csv,
          look_alike_pairs.csv, font_distances.csv
USED BY   data_figures, hasy (meta.json), training/ (all splits)
RUN       python src/data_collection/make_dataset.py
------------------------------------------------------------------------------

Build the stage-1 dataset (printed single symbols), following the proposal.

  python src/data_collection/make_dataset.py --fonts-dir fonts --out data

Steps
  1. find the fonts in fonts/ and their split (fonts_config.py; split by font)
  2. coverage check: a (class, style, font) exists only if the font maps the
     codepoint AND the render is non-empty (failed / empty renders discarded)
  3. per font: clean render of every available (class, style); train fonts also
     get N_TRAIN augmented samples per class, val/test fonts get N_EVAL augmented
     samples per class for a robustness check (flag aug = 1)
  4. leakage guard: fonts in different splits must not be the same family
     (font-to-font distance >= LEAK_TOL); writes data/font_distances.csv
  5. merge classes whose clean renders are (near) pixel-identical in every font
  6. save data/{train,val,test}.npz, data/clean.npz, data/meta.json,
     data/class_counts.csv, data/look_alike_pairs.csv

Everything is seeded: per-font generators are derived from (seed, font index),
so the output does not depend on the number of worker processes.
"""
import argparse, csv, json, os, re, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import augment as A
from fonts_config import FONTS, LEAK_TOL, SPLITS
from render import OUT, CLEAN_SIDE, RENDER_PX, cmap, normalize_clean, render_hi
from symbols import CLASSES, STYLE_PROB_DEFAULT

STYLE_ID = {"upright": 0, "italic": 1}
MERGE_TOL = 0.10   # normalized pixel difference must be below this in EVERY shared font


def find_fonts(fonts_dir):
    """Fonts from fonts_config.FONTS that are present in fonts_dir, with split and family."""
    files = sorted(Path(fonts_dir).iterdir()) if Path(fonts_dir).exists() else []
    found = []
    for name, (_, rx, split, family, _) in FONTS.items():
        hit = [f for f in files if re.search(rx, f.name, re.I)]
        if hit:
            found.append(dict(name=name, path=str(hit[0]), split=split, family=family))
    return found


def font_job(args):
    """All images of ONE font (runs in a worker process): for every class and style the
    font has, one clean render, plus augmented copies (N_TRAIN per class for train fonts,
    N_EVAL for val/test fonts). Also returns the clean render per class for later checks."""
    fi, font, seed, n_train, n_eval = args
    rng = np.random.default_rng([seed, fi])          # seeded per font -> same output for any #workers
    have = cmap(font["path"])
    X, y, sty, aug = [], [], [], []
    clean = np.zeros((len(CLASSES), OUT, OUT), np.uint8)
    clean_ok = np.zeros(len(CLASSES), bool)
    cover, failed = {}, []
    for ci, c in enumerate(CLASSES):
        hi = {}
        for s, cp in c["styles"].items():
            if cp not in have:
                continue
            r = render_hi(font["path"], cp)
            if r is None:
                failed.append(f"{c['latex']}:{s}")
            else:
                hi[s] = r
        cover[c["latex"]] = sorted(hi)
        if not hi:
            continue
        # clean render: LaTeX's default style if available
        d = c["default"] if c["default"] in hi else next(iter(hi))
        clean[ci], clean_ok[ci] = normalize_clean(hi[d]), True
        for s, h in hi.items():                                  # one clean sample per style
            X.append(normalize_clean(h)); y.append(ci); sty.append(STYLE_ID[s]); aug.append(0)
        n_aug = (n_train - len(hi)) if font["split"] == "train" else n_eval
        styles = list(hi)
        for _ in range(max(n_aug, 0)):
            if len(styles) == 1:
                s = styles[0]
            else:
                other = [t for t in styles if t != d][0]
                s = d if rng.random() < STYLE_PROB_DEFAULT else other
            X.append(A.augment(hi[s], rng)); y.append(ci); sty.append(STYLE_ID[s]); aug.append(1)
    return dict(fi=fi, X=np.stack(X), y=np.array(y, np.int16), style=np.array(sty, np.int8),
                aug=np.array(aug, np.int8), clean=clean, clean_ok=clean_ok,
                cover=cover, failed=failed)


def pair_distances(clean, ok):
    """D[f, a, b] = sum|clean_a - clean_b| / sum max(clean_a, clean_b) in font f:
    0 = pixel-identical, 1 = no overlapping ink (nan if either glyph is missing).
    Normalizing by ink keeps thin glyphs (-, |) from looking similar to everything."""
    F, C = ok.shape
    x = clean.reshape(F, C, -1).astype(np.float32) / 255
    D = np.full((F, C, C), np.nan, np.float32)
    for f in range(F):
        a, b = x[f][:, None, :], x[f][None, :, :]
        d = np.abs(a - b).sum(-1) / np.maximum(np.maximum(a, b).sum(-1), 1e-6)
        m = ok[f][:, None] & ok[f][None, :]
        D[f][m] = d[m]
    return D


def font_distances(clean, ok):
    """F x F matrix: mean over shared classes of the normalized pixel difference."""
    F = len(ok)
    x = clean.astype(np.float32) / 255
    S = np.zeros((F, F), np.float32)
    for p in range(F):
        for q in range(p + 1, F):
            m = ok[p] & ok[q]
            a, b = x[p][m], x[q][m]
            d = np.abs(a - b).sum((1, 2)) / np.maximum(np.maximum(a, b).sum((1, 2)), 1e-6)
            S[p, q] = S[q, p] = d.mean()
    return S


def main():
    """Steps 1-6 of the module docstring; each step is marked with a # ---- comment."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonts-dir", default="fonts")
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=360)
    ap.add_argument("--n-train", type=int, default=60, help="images per (class, train font)")
    ap.add_argument("--n-eval", type=int, default=10, help="augmented images per (class, val/test font)")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--allow-leak", action="store_true",
                    help="build the dataset even if two fonts in different splits are the same family")
    a = ap.parse_args()
    t0 = time.time()

    # ---------------- find fonts and render them (one process per font)
    fonts = find_fonts(a.fonts_dir)
    if not fonts:
        sys.exit(f"no fonts found in {a.fonts_dir}/ (run src/data_collection/get_fonts.py first)")
    for s in SPLITS:
        names = [f["name"] for f in fonts if f["split"] == s]
        print(f"{s:5s}: {len(names)} fonts {names}")
        if not names:
            sys.exit(f"split '{s}' has no fonts")

    jobs = [(fi, f, a.seed, a.n_train, a.n_eval) for fi, f in enumerate(fonts)]
    res = [None] * len(fonts)
    with Pool(max(1, min(a.workers, len(fonts)))) as pool:
        for r in pool.imap_unordered(font_job, jobs):
            res[r["fi"]] = r
            print(f"  rendered {fonts[r['fi']]['name']:12s} {len(r['y']):6d} images "
                  f"({time.time() - t0:.0f}s)")

    clean = np.stack([r["clean"] for r in res]); ok = np.stack([r["clean_ok"] for r in res])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    # ---------------- leakage guard: no font family across splits
    FD = font_distances(clean, ok)
    names = [f["name"] for f in fonts]
    with open(out / "font_distances.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["font", "split"] + names)
        for p, f in enumerate(fonts):
            w.writerow([f["name"], f["split"]] + [round(float(v), 3) for v in FD[p]])
    leaks = sorted((float(FD[p, q]), names[p], fonts[p]["split"], names[q], fonts[q]["split"])
                   for p in range(len(fonts)) for q in range(p + 1, len(fonts))
                   if fonts[p]["split"] != fonts[q]["split"] and FD[p, q] < LEAK_TOL)
    cross = min(float(FD[p, q]) for p in range(len(fonts)) for q in range(len(fonts))
                if fonts[p]["split"] != fonts[q]["split"])
    print(f"\nclosest fonts in different splits: distance {cross:.2f} (leak threshold {LEAK_TOL})")
    for d, n1, s1, n2, s2 in leaks:
        print(f"  LEAK: {n1} ({s1}) ~ {n2} ({s2}): {d:.2f}")
    if leaks and not a.allow_leak:
        sys.exit("same-family fonts in different splits: move them into one split in "
                 "fonts_config.py (see data/font_distances.csv), or pass --allow-leak")

    # ---------------- merge near pixel-identical classes (proposal: cleaning step)
    D = pair_distances(clean, ok)
    with np.errstate(all="ignore"):
        dmax = np.nanmax(D, 0); dmean = np.nanmean(D, 0); nshared = (~np.isnan(D)).sum(0)
    C = len(CLASSES)
    parent = list(range(C))
    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i
    merges = []
    for i in range(C):
        for j in range(i + 1, C):
            if nshared[i, j] >= 2 and dmax[i, j] < MERGE_TOL:
                ri, rj = root(i), root(j)
                if ri != rj:
                    parent[rj] = ri
                    merges.append(dict(keep=CLASSES[ri]["latex"], merged=CLASSES[j]["latex"],
                                       max_diff=round(float(dmax[i, j]), 4)))
    roots = sorted({root(i) for i in range(C)})
    new_id = {r: k for k, r in enumerate(roots)}
    remap = np.array([new_id[root(i)] for i in range(C)], np.int16)
    classes_out = [dict(latex=CLASSES[r]["latex"], group=CLASSES[r]["group"],
                        aliases=[CLASSES[i]["latex"] for i in range(C) if root(i) == r and i != r])
                   for r in roots]

    # ---------------- assemble splits (concatenate fonts, apply the class merge, save)
    K = len(classes_out)
    counts = {s: np.zeros(K, int) for s in SPLITS}
    stats = {}
    for s in SPLITS:
        rs = [r for r in res if fonts[r["fi"]]["split"] == s]
        X = np.concatenate([r["X"] for r in rs])
        y = remap[np.concatenate([r["y"] for r in rs])]
        font = np.concatenate([np.full(len(r["y"]), r["fi"], np.int8) for r in rs])
        style = np.concatenate([r["style"] for r in rs])
        aug = np.concatenate([r["aug"] for r in rs])
        np.savez_compressed(out / f"{s}.npz", X=X, y=y, font=font, style=style, aug=aug)
        counts[s] = np.bincount(y, minlength=K)
        stats[s] = dict(images=int(len(y)), clean=int((aug == 0).sum()),
                        augmented=int((aug == 1).sum()),
                        fonts=[fonts[r["fi"]]["name"] for r in rs])
        if s == "train":
            stats[s]["pixel_mean"] = float(X.mean() / 255)
            stats[s]["pixel_std"] = float(X.std() / 255)
    np.savez_compressed(out / "clean.npz", clean=clean, ok=ok)

    # ---------------- statistics files for the report
    with open(out / "class_counts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["id", "latex", "group", "aliases"] + SPLITS)
        for k, c in enumerate(classes_out):
            w.writerow([k, c["latex"], c["group"], " ".join(c["aliases"])] + [counts[s][k] for s in SPLITS])

    pairs = [(float(dmean[i, j]), i, j) for i in range(C) for j in range(i + 1, C)
             if nshared[i, j] >= 2 and root(i) != root(j)]
    pairs.sort()
    with open(out / "look_alike_pairs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["a", "b", "mean_diff", "max_diff", "fonts_shared"])
        for d, i, j in pairs[:60]:
            w.writerow([CLASSES[i]["latex"], CLASSES[j]["latex"], round(d, 4),
                        round(float(dmax[i, j]), 4), int(nshared[i, j])])

    # ---------------- meta.json: everything needed to reproduce / interpret the data
    meta = dict(
        seed=a.seed, n_train=a.n_train, n_eval=a.n_eval, image_size=OUT,
        clean_glyph_side=CLEAN_SIDE, render_px=RENDER_PX, style_prob_default=STYLE_PROB_DEFAULT,
        merge_tol=MERGE_TOL,
        augment=dict(rot_deg=A.ROT_DEG, side=A.SIDE, max_shift=A.MAX_SHIFT, p_stroke=A.P_STROKE,
                     p_blur=A.P_BLUR, blur_r=A.BLUR_R, p_noise=A.P_NOISE, noise_sd=A.NOISE_SD,
                     p_thicken=A.P_THICKEN, p_elastic=A.P_ELASTIC, elastic_grid=A.ELASTIC_GRID,
                     elastic_amp=A.ELASTIC_AMP, p_slant=A.P_SLANT, slant=A.SLANT,
                     p_pen=A.P_PEN, pen_widths=A.PEN_WIDTHS, pen_p=A.PEN_P, p_binarize=A.P_BINARIZE),
        classes=classes_out, original_classes=[c["latex"] for c in CLASSES],
        remap=remap.tolist(), merges=merges,
        fonts=[dict(f, path=Path(f["path"]).name,
                    covered=sum(bool(v) for v in r["cover"].values()),
                    missing=[k for k, v in r["cover"].items() if not v],
                    failed_renders=r["failed"]) for f, r in zip(fonts, res)],
        splits=stats, min_cross_split_font_distance=round(cross, 3), leak_tol=LEAK_TOL)
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)

    print(f"\n{C} classes -> {K} after merging {len(merges)}: "
          + ", ".join(f"{m['merged']}->{m['keep']}" for m in merges))
    for s in SPLITS:
        print(f"{s:5s}: {stats[s]['images']:7d} images, per-class min/max "
              f"{counts[s].min()}/{counts[s].max()}")
    for fm in meta["fonts"]:
        if fm["missing"]:
            print(f"  {fm['name']}: missing {len(fm['missing'])} classes")
    print(f"done in {time.time() - t0:.0f}s -> {out}/")


if __name__ == "__main__":
    main()
