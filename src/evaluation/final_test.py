r"""final_test.py: step 7, evaluate the final models ONCE on the held-out test data.

PIPELINE  data_collection/: 1 get_fonts -> 2 make_dataset -> 3 data_figures -> 4 hasy
          training/:        5 tune (optional) -> 6 train
          evaluation/:      7 final_test (once, at the end)
THIS FILE step 7
READS     results/<model>.pt|.json and results/<model>_ft.pt (from train.py),
          data/test.npz (4 unseen test fonts), data/hasy.npz (HASY test writers)
WRITES    results/final_test.json, figures/test_table.tex, figures/test_macros.tex
USED BY   the final report
RUN       python src/evaluation/final_test.py [--models ann cnn]
------------------------------------------------------------------------------

Run this only after every choice (architecture, hyperparameters, epochs) has been made
on validation data, as in Lab 3 part 3(d). Nothing is trained here: the saved best
checkpoints are loaded and evaluated on
  printed test   clean images of the 4 test fonts (+ augmented copies)
  handwriting    HASY test writers (no writer overlaps with the val writers)
for the trained models and, if they exist, the fine-tuned ones.
"""
import argparse, json, os, sys
from pathlib import Path

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(SRC, d) for d in ("data_collection", "training", "evaluation")]

from data import load_data
from metrics import evaluate_hasy, evaluate_printed, group_map, load_trained
from models import MODELS
from report import test_macros, test_table


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--results", default="results")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--models", nargs="+", default=["ann", "cnn"], choices=list(MODELS))
    return ap.parse_args()


def main():
    a = get_args()
    results, figs = Path(a.results), Path(a.figures)
    figs.mkdir(parents=True, exist_ok=True)
    print("FINAL TEST: run once, after all choices were made on validation data.")

    first = json.loads((results / f"{a.models[0]}.json").read_text(encoding="utf-8"))
    seed = first["config"]["seed"]                     # same HASY writer split as training
    d = load_data(a.data, seed, splits=("test",))
    K, names, fonts, hasy = d["K"], d["names"], d["fonts"], d["hasy"]
    G = group_map(names)
    nan = dict(acc=float("nan"), acc_merged=float("nan"))

    res = {}
    for name in a.models:
        net, record = load_trained(name, K, results)
        r = {}
        r["printed_test"], _ = evaluate_printed(net, *d["test"], K, G, fonts)
        r["hasy_test"] = evaluate_hasy(net, *hasy["test"], K, G)[0] if hasy is not None else nan
        # fine-tuned model: only if the last train.py run fine-tuned it (else <name>_ft.pt is stale)
        if "hasy_ft" in record and (results / f"{name}_ft.pt").exists() and hasy is not None:
            ft, _ = load_trained(name, K, results, suffix="_ft")
            r["ft"] = dict(printed_test=evaluate_printed(ft, *d["test"], K, G, fonts)[0],
                           hasy_test=evaluate_hasy(ft, *hasy["test"], K, G)[0])
        res[name] = r
        p, h = r["printed_test"]["clean"], r["hasy_test"]
        print(f"{name}: printed test acc {p['acc']:.4f} (merged {p['acc_merged']:.4f}, macro-F1 "
              f"{p['macro_f1']:.4f}) | handwriting test acc {h['acc']:.4f} (merged {h['acc_merged']:.4f})")
        print("  per test font:", {k: round(x, 3) for k, x in r["printed_test"]["per_font_clean_acc"].items()})
        if "ft" in r:
            print(f"  fine-tuned: handwriting test acc {r['ft']['hasy_test']['acc']:.4f}, "
                  f"printed test acc {r['ft']['printed_test']['clean']['acc']:.4f}")

    (results / "final_test.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    test_table(res, figs)
    test_macros(res, figs)
    print(f"\nwrote {results}/final_test.json, {figs}/test_table.tex, {figs}/test_macros.tex")


if __name__ == "__main__":
    main()
