# Data generation and processing

How every dataset in this project is built, in the order the scripts run. All scripts live in
`src/data_collection/` (stage-2 evaluation helpers in `src/evaluation/`), run from the project root, are
seeded (default seed 360), and write to `data/` and `figures/` (both regenerated, `data/` is not in git).

| Step | Script | Output | Time (CPU) |
|---|---|---|---|
| 1 | `get_fonts.py` | `fonts/*.otf` (18 math fonts) | ~1 min (download) |
| 2 | `make_dataset.py` | printed single symbols: `data/{train,val,test}.npz`, `clean.npz`, `meta.json`, statistics CSVs | ~2 min |
| 2b | `make_equations.py` | printed equations: `data/equations/` (stretch goal) | ~15 s |
| 3 | `data_figures.py` | dataset figures and tables in `figures/`, QA sheets in `figures/qa/` | ~1 min |
| 4 | `hasy.py` | handwritten symbols: `data/hasy.npz`, `data/hasy_mapping.csv` | ~1 min |

---

## 1. Classes (`symbols.py`)

- **160 classes**, one canonical LaTeX command per visual glyph (proposal, Appendix A):
  digits (10), Latin letters (52), Greek lowercase (29), Greek capitals that differ from Latin (11),
  arithmetic and relations (21), sets / logic / arrows / misc (24), punctuation and delimiters (13).
- Each class stores its LaTeX label, its group, and the Unicode code points of its **upright** and
  **math-italic** forms (letters and lowercase Greek have both; math italic uses the Unicode
  Mathematical Alphanumeric block, e.g. italic *h* is U+210E).
- **Near-identical groups** (`NEAR_IDENTICAL`, used only for the "merged" metrics): same shape up to size
  or position once cropped to the ink box, plus the look-alikes named in the proposal:
  `o/O/0/\circ`, `l/I/1/|`, `c/C`, `s/S`, `w/W`, `z/Z`, `v/V/\vee`, `\wedge/\Lambda`, `x/X/\times`,
  `./\cdot`, `\epsilon/\varepsilon/\in`, `\phi/\varphi`.

## 2. Fonts and the split (`fonts_config.py`, `get_fonts.py`)

- 18 free OpenType math fonts from CTAN / TeX Live (the fonts LaTeX itself uses through `unicode-math`).
  `get_fonts.py` copies each from a local TeX install if present, else downloads it from CTAN
  (several mirrors, 3 tries).
- **Split by font family**, so no family appears in two splits:

  | Split | Fonts |
  |---|---|
  | train (11) | LatinModern, NewCM, Termes, STIXTwo, XITS, Bonum, DejaVu, Libertinus, Garamond, Euler, Erewhon |
  | val (3) | KpMath, Asana, Pagella (all Palatino-like) |
  | test (4) | Schola (serif), Concrete (Computer-Modern-like), LeteSans, FiraMath (sans) |

- **Leakage guard:** font similarity = mean normalized pixel difference of the clean renders over all
  shared classes (same family: 0.02–0.33). `make_dataset.py` refuses to build if two fonts in different
  splits are closer than `LEAK_TOL = 0.34` (closest cross-split pair: 0.36). Erewhon (Utopia) is 0.34
  from STIX Two, so it is kept in train with the Times-like fonts.

## 3. Printed single symbols (`make_dataset.py`, `render.py`, `augment.py`)

Per font (one worker process per font; each font has its own seeded generator `(seed, font index)`,
so the output does not depend on the number of processes):

1. **Coverage check.** A (class, style, font) is kept only if the font maps the code point and the render
   has at least `MIN_INK = 20` ink pixels. Missing: Euler `\varrho`, `\varsigma`; FiraMath `\perp`
   (3 of 2,880 class-font pairs).
2. **Render.** Pillow draws the glyph at a 128 px em (`RENDER_PX`) on a large canvas, cropped to the ink
   box plus a margin (so later rotation never clips).
3. **Normalize (clean image).** Crop to the ink bounding box → pad to a square (glyph centred) →
   area-resize (box filter) so the longest side is 28 px (`CLEAN_SIDE`) → centre in a 32×32 canvas.
   Images are `uint8`, ink = 255 on 0. The loader later divides by 255 and standardizes with the training
   mean / std (stored in `meta.json`).
4. **Styles.** For letters and lowercase Greek, the math-italic form is chosen 70 % of the time
   (`STYLE_PROB_DEFAULT`, LaTeX's default), upright 30 % (`\mathrm`, `upgreek`).
5. **Augmentation** (training fonts: 60 images per class, `--n-train`; val / test fonts: 10 per class,
   `--n-eval`, used only as a robustness check). Applied to the high-resolution glyph, in this order:

   | Step | Probability | Setting |
   |---|---|---|
   | stroke weight | 0.5 | thicken (max-filter 3 or 5 px) 60 %, else thin (min-filter 3 px, skipped if > 40 % of the ink would vanish) |
   | elastic warp | 0.6 | 4×4 cell grid, inner corners moved by up to 5 % of the image size (Pillow MESH transform) |
   | slant | 0.5 | horizontal shear x += s·y, s uniform in [−0.3, 0.3] |
   | rotation | 1 | uniform in [−8°, 8°] |
   | scale + shift | 1 | longest side 22–30 px, shift up to ±2 px (kept inside the canvas) |
   | pen redraw | 0.35 | binarize at 128 → Zhang–Suen thinning to a 1 px skeleton → widen to 1 / 2 / 3 px (prob. 0.2 / 0.5 / 0.3) |
   | binarize | 0.15 (if no pen redraw) | hard threshold at 128 (crisp strokes, no anti-aliasing) |
   | blur | 0.4 | Gaussian, radius 0.3–0.9 px |
   | noise | 0.4 | Gaussian pixel noise, sd 2–8 % of 255 |

   Elastic warp, slant, pen redraw and binarize imitate handwriting (thin, even, wobbly strokes); the rest
   follows the proposal.
6. **Merge identical classes.** Two classes would be merged if their clean renders differ by less than
   `MERGE_TOL = 0.10` (normalized pixel difference) in **every** font they share; a tolerance is needed
   because anti-aliasing makes exact equality impossible. Nothing merges (closest pair: `\cdot` vs `.`).

**Result**

| Split | Fonts | Clean | Augmented | Total |
|---|---|---|---|---|
| train | 11 | 2,768 | 102,712 | 105,480 |
| val | 3 | 756 | 4,800 | 5,556 |
| test | 4 | 1,007 | 6,390 | 7,397 |

Classes are balanced: 600–660 training images and 3–6 clean validation images per class.

**Files** — `data/{train,val,test}.npz`:

| key | shape / dtype | meaning |
|---|---|---|
| `X` | (N, 32, 32) uint8 | ink = 255 on 0 |
| `y` | (N,) int16 | class id (`meta.json` → `classes`) |
| `font` | (N,) int8 | index into `meta.json` → `fonts` |
| `style` | (N,) int8 | 0 upright, 1 math italic |
| `aug` | (N,) int8 | 0 clean render, 1 augmented |

Also `clean.npz` (one clean render per font × class, for figures and checks), `meta.json` (classes,
fonts, split statistics, pixel mean / std, all augmentation settings), `class_counts.csv`,
`look_alike_pairs.csv` (most similar class pairs), `font_distances.csv`.

## 4. Handwritten symbols (`hasy.py`, split in `src/training/data.py`)

1. **Download** `HASYv2.tar.bz2` (34.6 MB, https://zenodo.org/records/259444) into `data/raw/` by hand.
2. **Map labels.** HASYv2 has 168k black-on-white 32×32 images in 369 classes with LaTeX labels.
   A label is kept if it equals one of my classes or one of these aliases: `\le→\leq`, `\ge→\geq`,
   `\ne→\neq`, `\rightarrow→\to`, `\gets→\leftarrow`, `\implies→\Rightarrow`, `\iff→\Leftrightarrow`,
   `\varnothing→\emptyset`, `\lnot→\neg`, `\land→\wedge`, `\lor→\vee`, `*→\ast`, `\mid→|`, `\vert→|`,
   `\lbrace→\{`, `\rbrace→\}`, `\colon→:`. Everything else is dropped.
   Result: 86,835 images, 146 of my 160 classes, 427 user ids.
3. **Normalize** exactly like the printed clean images: invert to ink = 255, crop to the ink box, pad to
   a square, area-resize to 28 px, centre in 32×32. Blank images are dropped.
4. **Writer split** (computed when the data is loaded, seed 360). HASY's official folds are random, so
   writers overlap; and one user id (16925) holds 76,159 images with no known writer, no images at all of
   24 classes (mostly digits and Latin letters), and a skewed class mix. So:
   - identified writers are shuffled and assigned whole: the first ~25 % of their images → **test**, the
     next ~25 % → **val**, the rest → **train**;
   - the bulk id → **train**.

   | Split | Images | Writers | Classes |
   |---|---|---|---|
   | train | 81,504 | bulk id + identified writers | 146 |
   | val | 2,735 | 198 | 142 |
   | test | 2,596 | 101 | — |

   Training images are imbalanced (20–2,877 per class; `\infty` most common), so fine-tuning draws
   class-balanced batches (`balanced_weights` in `data.py`). Validation writers are more even
   (no class above 2 %).

## 5. Printed equations — stage 2, stretch goal (`make_equations.py`)

**Rule:** each equation is one randomly chosen *statement template* whose slots are filled with random
*expressions*, built recursively from *atoms* joined by binary operators. Every token is one of the 160
classes. Only flat (single-line) math: no fractions, exponents, subscripts or roots. The grammar checks
form, not meaning (e.g. `Σ ⊇ (41, ∞)` can occur).

1. **Statement template**

   | Prob. | Template | Example |
   |---|---|---|
   | 45 % | expr REL expr | `2x + y = α` |
   | 8 % | expr (≤ 1 op) `<`/`\leq` var `<`/`\leq` expr (≤ 1 op) | `0 < x ≤ 5` |
   | 10 % | f `(` var `)` `=` expr, f ∈ {f, g, h, F, φ} | `f(x) = 3a − b` |
   | 8 % | `\forall`/`\exists` var SETREL Capital `,` expr REL expr | `∀x ∈ A, x > 0` |
   | 7 % | Capital `=` `\{` 2–4 vars separated by `,` `\}` | `A = {a, b, c}` |
   | 6 % | var SETREL `(`/`[` number `,` var or `\infty` `)`/`]` | `x ∈ [2, ∞)` |
   | 6 % | expr ARROW expr (≤ 1 op each) | `p ⇒ q` |
   | 4 % | `\neg` `(` expr `)` REL expr | `¬(p ∨ q) = r` |
   | 3 % | Capital `\cup`/`\cap` Capital `=`/`\neq` `\emptyset` | `A ∩ B = ∅` |
   | 3 % | `\angle` Capital `=`/`\approx` number | `∠A = 90` |

   Then, with probability 15 %, one of `.` `,` `;` `:` is appended.
2. **Expression** = atom, then 0–3 (uniform) more (BINOP, atom) pairs (inside function calls, absolute
   values and some templates at most 1).
3. **Atom**

   | Prob. | Atom | Example |
   |---|---|---|
   | 42 % | variable: Latin letter (65 %) or Greek letter (35 %) | `x`, `β` |
   | 18 % | number: 1 digit (50 %), 2 (33 %), 3 (17 %); no leading zero | `41` |
   | 10 % | number + variable | `2x` |
   | 6 % | two variables | `ab` |
   | 7 % | function call: f ∈ {f, g, h, φ, ψ}, `(` expr `)` | `f(x + 1)` |
   | 8 % | bracketed expr `( )`, `[ ]` or `\{ \}`, only at nesting depth < 2 | `(a − b)` |
   | 4 % | absolute value `\|` expr `\|` | `\|x\|` |
   | 2 % | `\partial` or `\nabla` + variable | `∂x` |
   | 1.5 % | factorial: variable `!` | `n!` |
   | 1.5 % | `\infty` or `\emptyset` | `∞` |

4. **Token pools** (repeats = weights)
   - BINOP: `+`×6, `-`×5, `\cdot`×2, `\times`×2, `/`, `\div`, `\pm`, `\mp`, `\ast`, `\circ`, `\cup`,
     `\cap`, `\wedge`, `\vee`
   - REL: `=`×8, `<`, `>`, `\leq`, `\geq` ×2 each, `\neq`, `\approx`, `\equiv`, `\sim`, `\propto`,
     `\ll`, `\gg`, `\to`, `\perp`
   - SETREL: `\in`, `\notin`, `\subset`, `\subseteq`, `\supset`, `\supseteq`
   - ARROW: `\Rightarrow`, `\Leftarrow`, `\Leftrightarrow`, `\to`, `\leftarrow`
5. **Length:** keep only equations with 3–20 tokens; otherwise draw a new one.

**Rendering**

- One font per equation, drawn from the split's fonts (train / val / test fonts as above); font size
  uniform in 40–56 px.
- Letters are math italic in 80 % of equations, upright in the rest (per equation, like real documents).
  If the font lacks a glyph, the whole equation is discarded and redrawn.
- Symbols are drawn left to right on one baseline (Pillow anchor `ls`) at their natural size, so
  relative size and height are kept (unlike the single-symbol crops).
- **Spacing** (TeX's rules, in units of 1/18 em) between consecutive tokens, by their classes:
  ordinary–ordinary 0; around binary operators 4; around relations 5; after punctuation 3.
  Each gap is scaled by a per-equation tightness (uniform 0.75–1.25) and a per-gap factor (0.6–1.4), and
  every advance gets an extra −0.03 to +0.05 em, so some neighbours touch (as in real documents).
- **Labels:** every symbol is also drawn on its own layer; its ink box (pixels > 96 of 255, the same
  threshold the segmenter uses) and class id are stored in reading order.

**Result:** train 2,000 equations (21,434 symbols), val 500 (5,238), test 500 (5,643); mean length
10.7 symbols; all 160 classes occur.

**Files** — `data/equations/`:
- `train/`, `val/`, `test/`: `eq_00000.png`, … (black ink on white)
- `<split>.jsonl`: one line per equation: `file`, `font`, `px`, `tokens` (LaTeX), `labels` (class ids),
  `boxes` (`[x0, y0, x1, y1]` per symbol)
- `summary.json`: counts, lengths, classes used per split

**Segmentation check** (`src/evaluation/segment.py`, no PyTorch needed):
ink > 96 → 8-connected components (components < 3 px dropped) → merge components whose horizontal
overlap is ≥ 50 % of the narrower one (the parts of `i`, `=`, `\leq`, `\div`, `:`, `;`, `!`) → sort left
to right. On the validation equations it finds 99.7 % of the symbols (IoU ≥ 0.5) and segments 93.6 % of
the equations exactly. Failures: `\ll` / `\gg` split into two pieces (two separate chevrons), and touching
neighbours merge.

## 6. Figures and checks (`data_figures.py`)

- `fig_fonts_grid`: clean renders of selected symbols in every font, grouped by split
- `fig_augmented`: clean render + augmented copies for a few classes
- `fig_coverage`: fraction of each symbol group each font covers
- `fig_class_counts`: training images per class
- `dataset_table.tex`: fonts and image counts per split
- `figures/qa/clean_<font>.png`: every class's clean render per font, for checking glyphs by eye

## Reproduce

```bash
python src/data_collection/get_fonts.py
python src/data_collection/make_dataset.py        # --n-train 60 --n-eval 10 --seed 360
python src/data_collection/make_equations.py      # optional (stage 2)
python src/data_collection/data_figures.py
python src/data_collection/hasy.py                # after downloading HASYv2 into data/raw/
python src/evaluation/segment.py                  # optional: segmentation check on val equations
```
