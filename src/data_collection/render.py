"""render.py: draw one symbol from one font and turn it into a 32x32 image.

PIPELINE  data_collection/: 1 get_fonts -> 2 make_dataset -> 3 data_figures -> 4 hasy
          training/:        5 tune (optional) -> 6 train
          evaluation/:      7 final_test (once, at the end)
THIS FILE helper: image functions
READS     a font file (.otf/.ttf)
WRITES    nothing (returns images)
USED BY   make_dataset (clean renders), augment (place/to_square), hasy (same normalization)
RUN       not run directly
------------------------------------------------------------------------------

Glyph rendering and the proposal's normalization:
render at high resolution -> crop to bounding box -> pad to square -> resize to
32x32 with area interpolation. Images are ink = 255 on background = 0 (uint8);
scaling to [0, 1] and standardizing happen in the training data loader.
"""
from functools import lru_cache

import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

RENDER_PX = 128      # font size of the high-resolution render (em = 128 px)
OUT = 32             # final image side
CLEAN_SIDE = 28      # longest glyph side inside the 32x32 image for clean renders
MIN_INK = 20         # renders with fewer ink pixels (at RENDER_PX) count as failed


@lru_cache(maxsize=None)
def cmap(path):
    """Set of codepoints the font maps (used for the coverage check)."""
    f = TTFont(path, lazy=True, fontNumber=0)
    cps = set(f.getBestCmap().keys())
    f.close()
    return frozenset(cps)


@lru_cache(maxsize=None)
def pil_font(path, size=RENDER_PX):
    """Cached PIL font object (each font is used for thousands of renders)."""
    return ImageFont.truetype(path, size)


def render_hi(path, cp):
    """High-resolution render of one codepoint, cropped to its bounding box plus a
    margin (so rotation does not clip). Returns an 'L' image or None if empty."""
    f = pil_font(path)
    S = RENDER_PX * 3                                  # big canvas so nothing is clipped
    im = Image.new("L", (S, S), 0)
    ImageDraw.Draw(im).text((S // 2, S // 2), chr(cp), font=f, fill=255, anchor="mm")
    bb = im.getbbox()
    if bb is None:                                     # font draws nothing for this codepoint
        return None
    if int((np.asarray(im) > 127).sum()) < MIN_INK:
        return None
    m = RENDER_PX // 4
    return im.crop((bb[0] - m, bb[1] - m, bb[2] + m, bb[3] + m))


def to_square(im):
    """Crop to the ink bounding box and pad to a square (glyph centred)."""
    bb = im.getbbox()
    c = im.crop(bb)
    s = max(c.size)
    sq = Image.new("L", (s, s), 0)
    sq.paste(c, ((s - c.size[0]) // 2, (s - c.size[1]) // 2))
    return sq


def place(sq, side=CLEAN_SIDE, dx=0, dy=0):
    """Resize the square glyph to side x side with area interpolation (BOX) and
    paste it into an OUT x OUT canvas, centred and shifted by (dx, dy)."""
    g = sq.resize((side, side), Image.Resampling.BOX)
    out = Image.new("L", (OUT, OUT), 0)
    x0 = (OUT - side) // 2 + dx
    y0 = (OUT - side) // 2 + dy
    out.paste(g, (x0, y0))
    return out


def normalize_clean(hi):
    """Clean (un-augmented) 32x32 image from a high-resolution render."""
    return np.asarray(place(to_square(hi)), dtype=np.uint8)
