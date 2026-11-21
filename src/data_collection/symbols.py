"""symbols.py: the 160 classes (what the model predicts) and the near-identical groups.

PIPELINE  data_collection/: 1 get_fonts -> 2 make_dataset -> 3 data_figures -> 4 hasy
          training/:        5 tune (optional) -> 6 train
          evaluation/:      7 final_test (once, at the end)
THIS FILE helper: defines labels; no computation
READS     nothing
WRITES    nothing
USED BY   make_dataset (rendering), data_figures, evaluation/metrics (NEAR_IDENTICAL)
RUN       python src/data_collection/symbols.py   (prints class counts as a sanity check)
------------------------------------------------------------------------------

Class table: one canonical LaTeX command per visual glyph (proposal, Appendix A).
The ORDER of CLASSES defines the class ids (0 = '0', ..., 159 = '/').

Each class has
  latex   : the label (canonical LaTeX command)
  group   : symbol group, used for statistics / figures
  styles  : dict style -> Unicode codepoint. "upright" is the plain codepoint,
            "italic" is the Mathematical Italic codepoint (how LaTeX typesets
            letters in math mode). A font that lacks a codepoint simply does not
            produce that (class, style) pair; see render.coverage().
  default : the style LaTeX uses by default in math mode (used for clean renders,
            QA sheets and near-duplicate merging)
"""

# ---------------------------------------------------------------- helpers
def _latin_italic(ch):
    """Unicode Mathematical Italic codepoint of a Latin letter (math-mode a..z, A..Z)."""
    if ch == "h":                       # math italic h lives in Letterlike Symbols
        return 0x210E
    if ch.islower():
        return 0x1D44E + ord(ch) - ord("a")
    return 0x1D434 + ord(ch) - ord("A")


def _greek_lower_italic(cp):
    """Math italic codepoint of a lowercase Greek letter."""
    # U+03B1..U+03C9 (alpha..omega incl. final sigma) -> U+1D6FC..U+1D714
    return 0x1D6FC + cp - 0x03B1


def _greek_upper_italic(cp):
    """Math italic codepoint of an uppercase Greek letter."""
    # U+0391..U+03A9 -> U+1D6E2..U+1D6FA
    return 0x1D6E2 + cp - 0x0391


# (latex, upright cp, italic cp) for the Greek variant symbols (Unicode "symbol" forms)
_GREEK_LOWER = [
    ("\\alpha", 0x03B1, None), ("\\beta", 0x03B2, None), ("\\gamma", 0x03B3, None),
    ("\\delta", 0x03B4, None),
    ("\\epsilon", 0x03F5, 0x1D716),     # lunate epsilon
    ("\\varepsilon", 0x03B5, None),     # "3"-shaped epsilon
    ("\\zeta", 0x03B6, None), ("\\eta", 0x03B7, None), ("\\theta", 0x03B8, None),
    ("\\vartheta", 0x03D1, 0x1D717),
    ("\\iota", 0x03B9, None), ("\\kappa", 0x03BA, None), ("\\lambda", 0x03BB, None),
    ("\\mu", 0x03BC, None), ("\\nu", 0x03BD, None), ("\\xi", 0x03BE, None),
    ("\\pi", 0x03C0, None),
    ("\\varpi", 0x03D6, 0x1D71B),
    ("\\rho", 0x03C1, None),
    ("\\varrho", 0x03F1, 0x1D71A),
    ("\\sigma", 0x03C3, None),
    ("\\varsigma", 0x03C2, None),
    ("\\tau", 0x03C4, None), ("\\upsilon", 0x03C5, None),
    ("\\phi", 0x03D5, 0x1D719),         # straight phi (unicode-math convention)
    ("\\varphi", 0x03C6, None),         # loopy phi
    ("\\chi", 0x03C7, None), ("\\psi", 0x03C8, None), ("\\omega", 0x03C9, None),
]

_GREEK_UPPER = [("\\Gamma", 0x0393), ("\\Delta", 0x0394), ("\\Theta", 0x0398),
                ("\\Lambda", 0x039B), ("\\Xi", 0x039E), ("\\Pi", 0x03A0),
                ("\\Sigma", 0x03A3), ("\\Upsilon", 0x03A5), ("\\Phi", 0x03A6),
                ("\\Psi", 0x03A8), ("\\Omega", 0x03A9)]

_ARITH_REL = [("+", 0x002B), ("-", 0x2212), ("=", 0x003D), ("<", 0x003C), (">", 0x003E),
              ("\\pm", 0x00B1), ("\\mp", 0x2213), ("\\times", 0x00D7), ("\\div", 0x00F7),
              ("\\cdot", 0x22C5), ("\\ast", 0x2217), ("\\circ", 0x2218),
              ("\\leq", 0x2264), ("\\geq", 0x2265), ("\\neq", 0x2260),
              ("\\approx", 0x2248), ("\\equiv", 0x2261), ("\\sim", 0x223C),
              ("\\propto", 0x221D), ("\\ll", 0x226A), ("\\gg", 0x226B)]

_SETS_MISC = [("\\in", 0x2208), ("\\notin", 0x2209), ("\\subset", 0x2282),
              ("\\subseteq", 0x2286), ("\\supset", 0x2283), ("\\supseteq", 0x2287),
              ("\\cup", 0x222A), ("\\cap", 0x2229), ("\\emptyset", 0x2205),
              ("\\forall", 0x2200), ("\\exists", 0x2203), ("\\neg", 0x00AC),
              ("\\wedge", 0x2227), ("\\vee", 0x2228), ("\\to", 0x2192),
              ("\\leftarrow", 0x2190), ("\\Rightarrow", 0x21D2), ("\\Leftarrow", 0x21D0),
              ("\\Leftrightarrow", 0x21D4), ("\\infty", 0x221E), ("\\partial", 0x2202),
              ("\\nabla", 0x2207), ("\\perp", 0x22A5), ("\\angle", 0x2220)]

_PUNCT = [(".", 0x002E), (",", 0x002C), (";", 0x003B), (":", 0x003A), ("!", 0x0021),
          ("(", 0x0028), (")", 0x0029), ("[", 0x005B), ("]", 0x005D),
          ("\\{", 0x007B), ("\\}", 0x007D), ("|", 0x007C), ("/", 0x002F)]


# ---------------------------------------------------------------- build table
def build_classes():
    """List of class dicts (latex, group, styles, default), in the fixed order
    digits, Latin, Greek lower, Greek upper, operators, sets/arrows, punctuation."""
    C = []
    add = lambda latex, group, styles, default: C.append(
        dict(latex=latex, group=group, styles=styles, default=default))

    for d in "0123456789":
        add(d, "digit", {"upright": ord(d)}, "upright")
    for ch in "abcdefghijklmnopqrstuvwxyz" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        add(ch, "latin", {"italic": _latin_italic(ch), "upright": ord(ch)}, "italic")
    for latex, up, it in _GREEK_LOWER:
        it = it if it is not None else _greek_lower_italic(up)
        add(latex, "greek_lower", {"italic": it, "upright": up}, "italic")
    for latex, up in _GREEK_UPPER:                  # LaTeX sets Greek capitals upright
        add(latex, "greek_upper", {"upright": up, "italic": _greek_upper_italic(up)}, "upright")
    for latex, cp in _ARITH_REL:
        add(latex, "arith_rel", {"upright": cp}, "upright")
    for latex, cp in _SETS_MISC:
        add(latex, "sets_logic_arrows", {"upright": cp}, "upright")
    for latex, cp in _PUNCT:
        add(latex, "punct_delim", {"upright": cp}, "upright")
    return C


# ---------------------------------------------------------------- public constants
CLASSES = build_classes()           # list of 160 dicts; index = class id before merging
GROUPS = ["digit", "latin", "greek_lower", "greek_upper",
          "arith_rel", "sets_logic_arrows", "punct_delim"]

# Probability of each style when a class has both (LaTeX default first).
# Letters appear upright in real documents too (\mathrm, \upalpha, text), so the
# non-default style is mixed in at a lower rate.
STYLE_PROB_DEFAULT = 0.7

# Near-identical groups, used only for evaluation ("pairs merged" metrics: a prediction
# anywhere in the true class's group counts as correct). Training is unchanged.
# Rule: (1) same shape up to size or position once the glyph is cropped to its ink box
# (case pairs, dot vs centred dot, o/0/circle, l/1/|), or (2) look-alikes named in the
# proposal (epsilon variants, phi variants, x vs times).
NEAR_IDENTICAL = [
    ["o", "O", "0", "\\circ"],
    ["l", "I", "1", "|"],
    ["c", "C"], ["s", "S"], ["w", "W"], ["z", "Z"],
    ["v", "V", "\\vee"], ["\\wedge", "\\Lambda"],
    ["x", "X", "\\times"],
    [".", "\\cdot"],
    ["\\epsilon", "\\varepsilon", "\\in"],
    ["\\phi", "\\varphi"],
]

if __name__ == "__main__":
    from collections import Counter
    print(len(CLASSES), "classes")
    print(Counter(c["group"] for c in CLASSES))
    assert len({c["latex"] for c in CLASSES}) == len(CLASSES)
