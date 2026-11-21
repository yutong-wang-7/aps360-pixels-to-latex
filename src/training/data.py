"""data.py: load the generated data as PyTorch datasets (printed fonts + HASYv2).

PIPELINE  data_collection/: 1 get_fonts -> 2 make_dataset -> 3 data_figures -> 4 hasy
          training/:        5 tune (optional) -> 6 train
          evaluation/:      7 final_test (once, at the end)
THIS FILE helper: data loading for training / evaluation
READS     data/{train,val,test}.npz, data/meta.json (step 2), data/hasy.npz (step 4)
WRITES    nothing
USED BY   training/tune.py, training/train.py, evaluation/ (metrics, final_test)
RUN       not run directly
------------------------------------------------------------------------------

Every split is returned as
  dataset  TensorDataset(X, y): X standardized float (N, 1, 32, 32), y int64 class ids
  info     numpy arrays for analysis and figures: y, font, aug, raw (uint8 images),
           and user (HASY writer id) for handwriting
Standardization uses the TRAIN pixel mean / std from meta.json for every split,
including handwriting, so all inputs are on the same scale.

HASYv2 split (writer-disjoint where writers are known, see hasy_split):
  identified writers, whole writers assigned: ~50 % of their images -> train,
                                              ~25 % -> val, ~25 % -> test
  images with no known writer (one bulk user id) -> train
The bulk id alone has no images of 24 classes (mostly digits / Latin letters) and is
dominated by a few math symbols, so identified writers are needed in train too.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

HASY_ANON = 16925   # HASY user id that holds most images (no known writer)
HASY_FRAC = dict(test=0.25, val=0.25)   # share of the identified writers' images (rest -> train)


def load_meta(data):
    """data/meta.json: classes, fonts, splits, train pixel mean / std."""
    return json.loads((Path(data) / "meta.json").read_text(encoding="utf-8"))


def to_tensor(X, mean, std):
    """uint8 images (N, 32, 32), ink = 255 -> standardized float tensor (N, 1, 32, 32)."""
    return torch.from_numpy(X).float().div_(255).sub_(mean).div_(std).unsqueeze(1)


def load_split(data, split, mean, std):
    """One printed split (train / val / test) -> (dataset, info)."""
    z = np.load(Path(data) / f"{split}.npz")
    y = z["y"].astype(np.int64)
    dataset = TensorDataset(to_tensor(z["X"], mean, std), torch.from_numpy(y))
    info = dict(y=y, font=z["font"].astype(int), aug=z["aug"].astype(int), raw=z["X"])
    return dataset, info


def subset(dataset, mask):
    """The images of a TensorDataset selected by a boolean numpy mask
    (e.g. only the clean validation images, aug == 0)."""
    m = torch.from_numpy(np.asarray(mask, bool))
    return TensorDataset(*(t[m] for t in dataset.tensors))


def hasy_split(user, seed):
    """Writer-disjoint split. Identified writers are shuffled and taken whole: the first
    ~25 % of their images -> test, the next ~25 % -> val, the rest -> train. Images
    without a known writer (HASY_ANON) -> train."""
    known = user != HASY_ANON
    u, c = np.unique(user[known], return_counts=True)
    o = np.random.default_rng(seed).permutation(len(u))
    share = np.cumsum(c[o]) / c.sum()            # share of identified images up to each writer
    t, v = HASY_FRAC["test"], HASY_FRAC["test"] + HASY_FRAC["val"]
    split = np.full(len(user), "train", dtype=object)
    split[np.isin(user, u[o][share <= t])] = "test"
    split[np.isin(user, u[o][(share > t) & (share <= v)])] = "val"
    return split


def load_hasy(data, mean, std, seed):
    """data/hasy.npz -> {"train" / "val" / "test": (dataset, info)}, or None if missing."""
    path = Path(data) / "hasy.npz"
    if not path.exists():
        return None
    z = np.load(path)
    split = hasy_split(z["user"], seed)
    out = {}
    for s in ("train", "val", "test"):
        m = split == s
        y = z["y"][m].astype(np.int64)
        dataset = TensorDataset(to_tensor(z["X"][m], mean, std), torch.from_numpy(y))
        info = dict(y=y, font=np.full(len(y), -1), aug=np.zeros(len(y), int),
                    raw=z["X"][m], user=z["user"][m])
        out[s] = (dataset, info)
    return out


def random_subset(dataset, frac, seed):
    """A random fraction of a dataset (fixed seed), e.g. to make tuning runs faster."""
    if frac >= 1:
        return dataset
    n = len(dataset)
    idx = np.random.default_rng(seed).permutation(n)[:int(frac * n)]
    m = np.zeros(n, bool); m[idx] = True
    return subset(dataset, m)


def load_data(data, seed, splits=("train", "val")):
    """Everything the training / evaluation scripts need, in one dict:
      meta, names (class labels), K, fonts,
      <split>: (dataset, info) for each printed split in `splits`,
      val_clean: clean validation images only (the model-selection set),
      hasy: {"train" / "val" / "test": (dataset, info)} or None."""
    meta = load_meta(data)
    mean, std = meta["splits"]["train"]["pixel_mean"], meta["splits"]["train"]["pixel_std"]
    d = dict(meta=meta, names=[c["latex"] for c in meta["classes"]], fonts=meta["fonts"])
    d["K"] = len(d["names"])
    for s in splits:
        d[s] = load_split(data, s, mean, std)
    if "val" in d:
        d["val_clean"] = subset(d["val"][0], d["val"][1]["aug"] == 0)
    d["hasy"] = load_hasy(data, mean, std, seed)
    return d


def balanced_weights(*labels):
    """Sampling weight per image for ConcatDataset([part 1, part 2, ...]) with the given
    label arrays: every part gets the same share of each epoch, and inside a part every
    class is equally likely (class-balanced sampling)."""
    w = []
    for y in labels:
        cnt = np.bincount(y)
        w.append(1.0 / (len(labels) * (cnt > 0).sum() * cnt[y]))
    return np.concatenate(w)


def get_loader(dataset, batch_size, shuffle=True, weights=None):
    """PyTorch data loader over a dataset (as in Lab 3). With `weights`, batches are
    drawn with replacement in proportion to the weights (WeightedRandomSampler)."""
    if weights is not None:
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                        num_samples=len(dataset), replacement=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
