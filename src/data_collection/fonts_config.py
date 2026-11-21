"""fonts_config.py: which 18 math fonts are used and which split each one is in.

PIPELINE  data_collection/: 1 get_fonts -> 2 make_dataset -> 3 data_figures -> 4 hasy
          training/:        5 tune (optional) -> 6 train
          evaluation/:      7 final_test (once, at the end)
THIS FILE helper: configuration only
READS     nothing
WRITES    nothing
USED BY   get_fonts (where to download), make_dataset (split + leak check)
RUN       not run directly; edit FONTS to move a font between splits, then rebuild the data
------------------------------------------------------------------------------

Math fonts and the font-level train/val/test split (proposal: split by font).

All fonts are free OpenType math fonts distributed with TeX Live (CTAN), so they
are the same fonts LaTeX (XeLaTeX/LuaLaTeX + unicode-math) typesets with.

Split rules
  * a font family never crosses a split (families measured from the renders:
    LM+NewCM 0.02, STIX+XITS+Termes 0.28-0.36, Kp+Asana+Pagella 0.23-0.33,
    LeteSans+Fira 0.31; see data/font_distances.csv) -> no leakage
  * test spans 3 different styles (serif, Computer-Modern-like, sans), so the
    "unseen font" number is a real generalization test
"""

# name: (ctan_package_path, filename regex, split, family note)
# name: (ctan_package_path, filename regex, split, family note, path of the .otf inside the package)
FONTS = {
    # ---------------- train (11 fonts, 8 families)
    "LatinModern": ("fonts/lm-math",            r"latinmodern-math\.otf$",       "train", "Computer Modern (LaTeX default)", "opentype/latinmodern-math.otf"),
    "NewCM":       ("fonts/newcomputermodern",  r"NewCMMath-Regular\.otf$",      "train", "Computer Modern",                 "otf/NewCMMath-Regular.otf"),
    "Termes":      ("fonts/tex-gyre-math",      r"texgyretermes-math\.otf$",     "train", "Times",                           "opentype/texgyretermes-math.otf"),
    "STIXTwo":     ("fonts/stix2-otf",          r"STIXTwoMath-Regular\.otf$",    "train", "Times",                           "STIXTwoMath-Regular.otf"),
    "XITS":        ("fonts/xits",               r"XITSMath-Regular\.otf$",       "train", "Times",                           "XITSMath-Regular.otf"),
    "Bonum":       ("fonts/tex-gyre-math",      r"texgyrebonum-math\.otf$",      "train", "Bookman",                         "opentype/texgyrebonum-math.otf"),
    "DejaVu":      ("fonts/tex-gyre-math",      r"(texgyredejavu-math\.otf|DejaVuMathTeXGyre\.ttf)$", "train", "DejaVu (sans-like)", "opentype/texgyredejavu-math.otf"),
    "Libertinus":  ("fonts/libertinus-fonts",   r"LibertinusMath-Regular\.otf$", "train", "Linux Libertine",                 "LibertinusMath-Regular.otf"),
    "Garamond":    ("fonts/garamond-math",      r"Garamond-Math\.otf$",          "train", "Garamond",                        "Garamond-Math.otf"),
    "Euler":       ("fonts/euler-math",         r"Euler-Math\.otf$",             "train", "Euler (upright, calligraphic)",   "Euler-Math.otf"),
    "Erewhon":     ("fonts/erewhon-math",       r"Erewhon-Math\.otf$",           "train", "Utopia (0.34 from STIX Two, so kept with the Times group)",                          "Erewhon-Math.otf"),
    # ---------------- val (3 fonts, 1 family: Palatino-like)
    "KpMath":      ("fonts/kpfonts-otf",        r"KpMath-Regular\.otf$",         "val",   "Palatino-like (Kp)",              "KpMath-Regular.otf"),
    "Asana":       ("fonts/Asana-Math",         r"Asana-Math\.otf$",             "val",   "Palatino-like",                   "Asana-Math.otf"),
    "Pagella":     ("fonts/tex-gyre-math",      r"texgyrepagella-math\.otf$",    "val",   "Palatino",                        "opentype/texgyrepagella-math.otf"),
    # ---------------- test (4 fonts, 3 families: serif, Computer-Modern-like, sans)
    "Schola":      ("fonts/tex-gyre-math",      r"texgyreschola-math\.otf$",     "test",  "Century Schoolbook",              "opentype/texgyreschola-math.otf"),
    "Concrete":    ("fonts/concmath-otf",       r"Concrete-Math\.otf$",          "test",  "Concrete",                        "Concrete-Math.otf"),
    "LeteSans":    ("fonts/lete-sans-math",     r"LeteSansMath\.otf$",           "test",  "sans",                            "LeteSansMath.otf"),
    "FiraMath":    ("fonts/firamath",           r"FiraMath-Regular\.otf$",       "test",  "Fira (sans)",                     "FiraMath-Regular.otf"),
}

# Leakage guard (make_dataset.py): two fonts in different splits must differ by at
# least this much (mean normalized pixel difference over all shared classes).
# Measured: same family 0.02-0.33, different families >= 0.36.
LEAK_TOL = 0.34

SPLITS = ["train", "val", "test"]
