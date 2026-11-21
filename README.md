# From Pixels to LaTeX: math symbol image → LaTeX command (APS360)

Solo APS360 course project (Alan Wang). Input: a 32x32 image of one math symbol; output: its LaTeX
command (160 classes). Models: ANN baseline and CNN primary, trained on rendered printed symbols
(18 math fonts, split by font family) and tested on unseen fonts and, zero-shot and after fine-tuning,
on handwritten symbols (HASYv2). Flat printed equations (stage 2) are a stretch goal for the final report.

## Current validation results (unseen fonts / writers)
| Model | Params | Printed acc | Printed (merged) | Augmented | Handwriting zero-shot | Handwriting fine-tuned |
|---|---|---|---|---|---|---|
| ANN (baseline) | 316k | 96.3 | 98.4 | 85.6 | 49.4 | 68.7 |
| CNN (primary)  | 692k | 98.0 | 98.9 | 94.1 | 63.3 | 78.4 |

"Merged": a prediction in the true class's near-identical group (e.g. o / O / 0) counts as correct.
All numbers come from `figures/results_macros.tex`, written by `src/training/train.py`.

## Quick start (Windows, from the project folder)
```powershell
python -m venv .venv; .venv\Scripts\Activate.ps1
.\setup_gpu.ps1                                   # CUDA build of PyTorch + requirements (NVIDIA GPU)
python src/data_collection/get_fonts.py
python src/data_collection/make_dataset.py
python src/data_collection/hasy.py                # after downloading HASYv2 into data/raw/ (see below)
python src/training/tune.py
python src/training/train.py --finetune
```

## Layout
```
src/
  data_collection/      steps 1-4: build the datasets (CPU, any machine)
    symbols.py          160 classes (proposal Appendix A) + NEAR_IDENTICAL groups
    fonts_config.py     18 OpenType math fonts and the font-family split (11 train / 3 val / 4 test)
    get_fonts.py        1  collects the fonts into fonts/ (local TeX install first, then CTAN)
    render.py           render at 128 px -> crop to bbox -> pad to square -> 32x32 area resize
    augment.py          rotation, scale, shift, stroke weight, blur, noise + handwriting-style
                        elastic warp, slant, pen-line redraw, binarize (seeded)
    make_dataset.py     2  builds data/*.npz + meta.json + class_counts / look_alike_pairs / font_distances
    make_equations.py   2b flat equations (grammar, one font per equation) -> data/equations/ (stretch goal)
    data_figures.py     3  dataset figures (figures/) + per-font QA sheets (figures/qa/)
    hasy.py             4  HASYv2 handwritten symbols -> my classes, same normalization -> data/hasy.npz
  training/             steps 5-6: train the models (GPU if available)
    data.py             load datasets (printed splits, HASYv2 writer-disjoint split), get_loader
    models.py           ANN baseline, CNN primary (Lab 3 style), default sizes, build_model
    train_loop.py       train_net / evaluate / get_model_name (Lab 3 template; Adam + cosine)
    tune.py             5  hyperparameter grid search -> results/tuning/best_<model>.json
    train.py            6  train ANN + CNN with the tuned settings, validation results, fine-tuning
  evaluation/           metrics, figures, tables; step 7
    metrics.py          accuracy, top-5, macro-F1, pairs-merged metrics
    plotting.py         plot_training_curve (Lab 3), report figures
    report.py           LaTeX tables + macros for the report
    final_test.py       7  evaluate the final models ONCE on test fonts + HASY test writers
    segment.py          stage 2 (stretch): connected components + overlap merge; checks itself vs true boxes
    equation_eval.py    stage 2 (stretch): segment -> classify -> exact match (val fonts; --test at the end)
report/                 progress_report.tex (numbers fill in from figures/*_macros.tex)
fonts/  data/  results/  figures/qa/   generated, not in git
```

## Data generation (CPU, any machine, ~1-3 min)
```bash
pip install -r requirements.txt
python src/data_collection/get_fonts.py      # prints which fonts were found / downloaded
python src/data_collection/make_dataset.py   # defaults: --n-train 60 --n-eval 10 --seed 360
python src/data_collection/make_equations.py # stage 2 (stretch goal): 2000 / 500 / 500 equations
python src/data_collection/data_figures.py
```

HASYv2 (stage 3, handwriting): download `HASYv2.tar.bz2` (34.6 MB) from
https://zenodo.org/records/259444 into `data/raw/`, then `python src/data_collection/hasy.py`.

## Data format
`data/{train,val,test}.npz`
| key | shape / dtype | meaning |
|---|---|---|
| `X` | (N, 32, 32) uint8 | ink = 255 on 0; divide by 255, then standardize with `meta.json` `splits.train.pixel_mean/std` |
| `y` | (N,) int16 | class id, see `meta.json` `classes` (after merging) |
| `font` | (N,) int8 | index into `meta.json` `fonts` |
| `style` | (N,) int8 | 0 upright, 1 math italic |
| `aug` | (N,) int8 | 0 clean render, 1 augmented |

Val/test: report main metrics on the clean images (`aug == 0`, the proposal's setup);
the augmented copies (`aug == 1`) are a robustness check.

## Decisions (vs. proposal)
- Same pipeline as the proposal. More fonts than "roughly a dozen" (18) since TeX Live has many.
- Letters/Greek lowercase: math italic 70 %, upright 30 % (LaTeX default first; \mathrm / \upalpha also occur).
- Split by font family, not just by font: a family never crosses a split. Families were measured
  from the renders (mean normalized pixel difference over all classes; same family 0.02-0.33,
  different families >= 0.36). `make_dataset.py` refuses to build if two fonts in different splits
  are closer than `LEAK_TOL = 0.34` (override: `--allow-leak`).
  - train (11 fonts, 8 families): LatinModern, NewCM, Termes, STIXTwo, XITS, Bonum, DejaVu, Libertinus, Garamond, Erewhon, Euler
    (Erewhon/Utopia is 0.34 from STIX Two, right on the threshold, so it stays with the Times group in train)
  - val (3 fonts, 1 family): KpMath, Asana, Pagella (Palatino-like)
  - test (4 fonts, 3 families): Schola (serif), Concrete (Computer-Modern-like), LeteSans + FiraMath (sans)
- Merge rule "pixel-identical within every font" uses a tolerance (normalized diff < 0.10 in every font)
  because anti-aliasing makes exact equality impossible. With the full font set nothing merges:
  `.` and `\cdot` differ in shape in 13 of 17 fonts, so they stay separate and form a look-alike pair.
- Coverage gaps (classes skipped for that font): Euler has no `\varrho`, `\varsigma`; FiraMath has no `\perp`.

## Training and evaluation (local; GPU if available, otherwise CPU)
GPU (Windows, NVIDIA): `.\setup_gpu.ps1` (in a PowerShell terminal) installs the CUDA build of
PyTorch matching your driver into `.venv`; then run everything with `.venv\Scripts\python`.
The training scripts use the GPU automatically when `torch.cuda.is_available()`.
```bash
python src/training/tune.py                  # grid search (lr, batch size, model size): 6 epochs on 30 % of train;
                                             #   score = mean of printed and handwritten val error
python src/training/train.py                 # ANN + CNN, 12 epochs, tuned size / lr / batch (else defaults), Adam, cosine
python src/training/train.py --load          # reuse results/<model>.pt, only re-evaluate (fast)
python src/training/train.py --finetune      # + fine-tune on printed + HASYv2 train (16 epochs, lr 3e-4, best epoch kept)
python src/evaluation/segment.py             # stage 2 (stretch) segmentation check (no PyTorch needed)
python src/evaluation/equation_eval.py       # stage 2 (stretch): whole equations with the trained ANN / CNN
python src/evaluation/final_test.py          # ONCE, at the end: test fonts + HASY test writers
```
`train.py` keeps the epoch with the lowest clean-val error (`results/<model>.pt`, `results/<model>.json`; every
epoch's checkpoint in `results/ckpt/`, curves as CSV in `results/curves/`) and writes
`figures/fig_curves`, `fig_errors`, `results_table.tex`, `confusions_table.tex`, `results_macros.tex`,
`near_identical.tex`, and with `data/hasy.npz` also `hasy_table.tex`, `fig_hasy`.
`tune.py` writes `figures/tuning_table.tex`, `fig_tuning`; `final_test.py` writes `test_table.tex`, `test_macros.tex`.
Selected models (tuning): ANN 1024-256-128-160 (316k parameters, lr 3e-3, batch 256);
CNN 4 conv blocks with 32-64-128-256 channels + FC 256 (692k parameters, lr 1e-3, batch 64).
Main metrics: top-1 / top-5 / macro-F1 on clean images of the unseen val fonts; augmented copies as a robustness check.

- **Pairs merged:** every metric is also computed with the near-identical groups in `src/data_collection/symbols.py`
  (`NEAR_IDENTICAL`) merged, i.e. a prediction anywhere in the true class's group counts as correct.
  Rule: same shape up to size/position after cropping to the ink box, plus the proposal's look-alikes.
- **Handwriting (HASYv2):** models trained on printed fonts are tested zero-shot on handwriting.
  HASY's official folds are random (writers overlap), and one user id holds 76k of the 87k mapped images
  with no known writer (and no images at all of 24 classes). Split: identified writers, whole writers,
  ~50 % of their images -> train, ~25 % -> val, ~25 % -> test; the bulk id -> train. Fine-tuning samples
  class-balanced batches, half printed and half handwritten.

## Report
`report/progress_report.tex` (course template) reads every number from `figures/*_macros.tex` and every
table from `figures/*.tex`, so re-running the scripts updates it; compile from inside `report/`.

## AI usage
Claude (Anthropic) was used as a coding and writing assistant: it helped implement parts of the code (in
particular data generation and the evaluation / plotting helpers), wrote most code comments, and helped
edit the report. The model and training-loop code follows the APS360 Lab 3 templates.
