r"""hasy.py: step 4, convert the HASYv2 handwriting dataset to my classes and format.

PIPELINE  data_collection/: 1 get_fonts -> 2 make_dataset -> 3 data_figures -> 4 hasy
          training/:        5 tune (optional) -> 6 train
          evaluation/:      7 final_test (once, at the end)
THIS FILE step 4 (needs data/meta.json from step 2)
READS     data/raw/HASYv2.tar.bz2 (download by hand), data/meta.json
WRITES    data/hasy.npz, data/hasy_mapping.csv, figures/hasy_macros.tex
USED BY   training/ and evaluation/ (handwriting zero-shot test and fine-tuning)
RUN       python src/data_collection/hasy.py
------------------------------------------------------------------------------

HASYv2 handwritten symbols -> my class set (proposal, stage 3).

HASYv2 (Thoma 2017, https://zenodo.org/records/259444) has 168k black-on-white
32x32 handwritten symbols in 369 classes, labelled with LaTeX commands. This
script keeps the ones whose label maps to one of my 160 classes and gives them
the same normalization as my printed data, so a model trained on printed fonts
can be tested zero-shot on handwriting (and later fine-tuned).

  1. download HASYv2.tar.bz2 (34.6 MB) from https://zenodo.org/records/259444
     into data/raw/
  2. python src/data_collection/hasy.py

Steps
  * read hasy-data-labels.csv (path, symbol_id, latex, user_id)
  * map the LaTeX label to my canonical command (exact match, plus aliases such
    as \le -> \leq, \rightarrow -> \to); unmapped HASY classes are dropped
  * invert to ink = 255 on 0, crop to the ink bounding box, pad to a square and
    resize so the longest side is 28 px in a 32x32 canvas (as in render.py)

Writes data/hasy.npz (X uint8, y int16, user int32), data/hasy_mapping.csv,
figures/hasy_macros.tex (\HasyImages, \HasyClasses, \HasyHasyClasses, \HasyUsers).
"""
import argparse, csv, io, json, os, sys, tarfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render import place, to_square

# HASY label -> my canonical label (only where the two spellings differ)
ALIASES = {
    "\\le": "\\leq", "\\ge": "\\geq", "\\ne": "\\neq",
    "\\rightarrow": "\\to", "\\gets": "\\leftarrow", "\\implies": "\\Rightarrow",
    "\\iff": "\\Leftrightarrow", "\\varnothing": "\\emptyset",
    "\\lnot": "\\neg", "\\land": "\\wedge", "\\lor": "\\vee", "*": "\\ast",
    "\\mid": "|", "\\vert": "|", "\\lbrace": "\\{", "\\rbrace": "\\}", "\\colon": ":",
}


def open_hasy(path):
    """Returns (rows of hasy-data-labels.csv, dict image file name -> PNG bytes).
    The archive is read once, sequentially (random access into .bz2 is slow)."""
    p = Path(path)
    if p.is_dir():
        lab = next(p.rglob("hasy-data-labels.csv"))
        rows = list(csv.DictReader(open(lab, encoding="utf-8")))
        imgs = {f.name: f.read_bytes() for f in p.rglob("*.png")}
        return rows, imgs
    rows, imgs = None, {}
    with tarfile.open(p, "r:*") as tf:
        for m in tf:
            if not m.isfile():
                continue
            if m.name.endswith("hasy-data-labels.csv"):
                rows = list(csv.DictReader(io.TextIOWrapper(tf.extractfile(m), encoding="utf-8")))
            elif m.name.endswith(".png"):
                imgs[Path(m.name).name] = tf.extractfile(m).read()
    if rows is None:
        sys.exit("hasy-data-labels.csv not found in the archive")
    return rows, imgs


def normalize(png):
    """PNG (black ink on white) -> 32x32 uint8 with ink = 255, cropped / padded / resized
    exactly like render.normalize_clean, so it matches the printed data; None if blank."""
    im = Image.open(io.BytesIO(png)).convert("L")
    im = Image.fromarray(255 - np.asarray(im, dtype=np.uint8))         # ink = 255 on 0
    if im.getbbox() is None:
        return None
    return np.asarray(place(to_square(im)), dtype=np.uint8)


def main():
    """Map each HASY label to my class (dropping the rest), normalize the images, save."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--hasy", default="data/raw/HASYv2.tar.bz2",
                    help="HASYv2.tar.bz2 or its extracted folder")
    ap.add_argument("--data", default="data")
    ap.add_argument("--figures", default="figures")
    a = ap.parse_args()
    data, figs = Path(a.data), Path(a.figures)
    if not Path(a.hasy).exists():
        sys.exit(f"{a.hasy} not found: download HASYv2.tar.bz2 from "
                 "https://zenodo.org/records/259444 into data/raw/")

    meta = json.loads((data / "meta.json").read_text(encoding="utf-8"))
    cid = {}                                      # my label (incl. merged aliases) -> class id
    for k, c in enumerate(meta["classes"]):
        cid[c["latex"]] = k
        for al in c.get("aliases", []):
            cid[al] = k

    rows, imgs = open_hasy(a.hasy)
    X, y, user, used, dropped, empty = [], [], [], Counter(), Counter(), 0
    for r in rows:
        lab = r["latex"].strip()
        mine = ALIASES.get(lab, lab)
        if mine is None or mine not in cid:
            dropped[lab] += 1
            continue
        png = imgs.get(Path(r["path"]).name)
        x = normalize(png) if png is not None else None
        if x is None:
            empty += 1
            continue
        X.append(x); y.append(cid[mine]); user.append(int(r.get("user_id") or -1))
        used[(mine, lab)] += 1
    if not X:
        sys.exit("no HASY image mapped to my classes; check the labels file")
    X = np.stack(X); y = np.array(y, np.int16); user = np.array(user, np.int32)
    np.savez_compressed(data / "hasy.npz", X=X, y=y, user=user)

    names = [c["latex"] for c in meta["classes"]]
    cnt = np.bincount(y, minlength=len(names))
    with open(data / "hasy_mapping.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["my_class", "hasy_labels", "images"])
        for k, n in enumerate(names):
            labs = sorted(h for (m, h) in used if cid[m] == k)
            w.writerow([n, " ".join(labs), int(cnt[k])])
    n_cls, n_hasy = int((cnt > 0).sum()), len({h for _, h in used})
    figs.mkdir(parents=True, exist_ok=True)
    (figs / "hasy_macros.tex").write_text(
        f"\\newcommand{{\\HasyImages}}{{{len(y):,}}}\n".replace(",", "{,}")
        + f"\\newcommand{{\\HasyClasses}}{{{n_cls}}}\n"
        + f"\\newcommand{{\\HasyHasyClasses}}{{{n_hasy}}}\n"
        + f"\\newcommand{{\\HasyUsers}}{{{len(set(user.tolist())):,}}}\n".replace(",", "{,}"),
        encoding="utf-8")
    print(f"HASYv2: {len(rows):,} images; kept {len(y):,} in {n_cls}/{len(names)} of my classes "
          f"(from {n_hasy} HASY classes, {len(set(user.tolist()))} writers); "
          f"dropped {sum(dropped.values()):,} from {len(dropped)} unmapped HASY classes; {empty} empty")
    missing = [n for n, c in zip(names, cnt) if c == 0]
    print(f"my classes with no HASY images ({len(missing)}): {' '.join(missing)}")


if __name__ == "__main__":
    main()
