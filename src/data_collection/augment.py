"""augment.py: random small distortions of a glyph (more varied training images).

PIPELINE  data_collection/: 1 get_fonts -> 2 make_dataset -> 3 data_figures -> 4 hasy
          training/:        5 tune (optional) -> 6 train
          evaluation/:      7 final_test (once, at the end)
THIS FILE helper: one function, augment(hi, rng)
READS     a high-resolution glyph from render.render_hi
WRITES    nothing (returns a 32x32 uint8 image)
USED BY   make_dataset
RUN       not run directly
------------------------------------------------------------------------------

Training augmentation (proposal: small random rotation, scale and shift,
stroke-weight change, blur, and noise). All randomness comes from the
numpy Generator passed in, so generation is fully seeded.

Handwriting-style additions (Nov 20, to narrow the print -> handwriting gap):
  elastic   smooth random warp of the glyph (wobbly strokes, uneven proportions)
  slant     horizontal shear, like slanted handwriting
  pen       redraw the glyph as a pen line: binarize -> thin to a 1 px skeleton ->
            widen to 1-3 px (handwritten strokes have one even width, print does not)
  binarize  hard threshold only: crisp strokes without anti-aliasing, as in HASYv2
"""
import numpy as np
from PIL import Image, ImageFilter

from render import OUT, place, to_square

# ranges (all "small", so the glyph identity never changes)
ROT_DEG = 8.0            # rotation uniform in [-8, 8] degrees
SIDE = (22, 30)          # scale: longest glyph side in the 32x32 image (clean = 28)
MAX_SHIFT = 2            # shift: up to +-2 px, kept inside the canvas
P_STROKE = 0.5           # stroke weight: thicken (3 or 5 px max-filter) or thin (3 px min-filter)
P_THICKEN = 0.6          # ... of which thicken (the rest thin)
P_ELASTIC = 0.6          # elastic warp: G x G control grid, corners moved up to AMP x glyph size
ELASTIC_GRID, ELASTIC_AMP = 4, 0.05
P_SLANT, SLANT = 0.5, 0.3   # shear x += s * y, s uniform in [-0.3, 0.3]
P_PEN = 0.35             # pen-line redraw (after resizing to 32x32)
PEN_WIDTHS, PEN_P = (1, 2, 3), (0.2, 0.5, 0.3)
P_BINARIZE = 0.15        # otherwise: threshold at 50 % after resizing
P_BLUR, BLUR_R = 0.4, (0.3, 0.9)
P_NOISE, NOISE_SD = 0.4, (0.02, 0.08)


def stroke(hi, rng):
    """Random stroke weight: with prob P_STROKE thicken (max-filter, 60 %) or thin
    (min-filter, 40 %); thinning is undone if it would remove > 40 % of the ink."""
    if rng.random() >= P_STROKE:
        return hi
    if rng.random() < P_THICKEN:                             # thicken
        return hi.filter(ImageFilter.MaxFilter(int(rng.choice([3, 3, 5]))))
    thin = hi.filter(ImageFilter.MinFilter(3))               # thin, but never erase hairlines
    a0 = (np.asarray(hi) > 127).sum()
    a1 = (np.asarray(thin) > 127).sum()
    return thin if a1 > 0.6 * a0 else hi


def skeleton(b):
    """Zhang-Suen thinning: boolean image -> 1 px wide strokes with the same shape."""
    b = b.astype(np.uint8)
    while True:
        changed = False
        for step in (0, 1):
            p = np.pad(b, 1)
            # neighbours N, NE, E, SE, S, SW, W, NW
            P2, P3, P4, P5 = p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:], p[2:, 2:]
            P6, P7, P8, P9 = p[2:, 1:-1], p[2:, :-2], p[1:-1, :-2], p[:-2, :-2]
            nb = [P2, P3, P4, P5, P6, P7, P8, P9, P2]
            B = sum(nb[:8])                                          # ink neighbours
            A = sum(((nb[k] == 0) & (nb[k + 1] == 1)).astype(np.uint8) for k in range(8))
            if step == 0:
                c = ((P2 * P4 * P6) == 0) & ((P4 * P6 * P8) == 0)
            else:
                c = ((P2 * P4 * P8) == 0) & ((P2 * P6 * P8) == 0)
            m = (b == 1) & (B >= 2) & (B <= 6) & (A == 1) & c
            if m.any():
                b[m] = 0; changed = True
        if not changed:
            return b.astype(bool)


def pen(im, rng):
    """Binarize, thin to a skeleton, then widen to a random even width (1, 2 or 3 px)."""
    sk = skeleton(np.asarray(im) >= 128)
    w = int(rng.choice(PEN_WIDTHS, p=PEN_P))
    out = sk.copy()
    if w >= 2:                                               # 2 px: add right / down / diagonal
        out[:, 1:] |= sk[:, :-1]; out[1:, :] |= sk[:-1, :]; out[1:, 1:] |= sk[:-1, :-1]
    if w == 3:                                               # 3 px: also left / up
        out[:, :-1] |= sk[:, 1:]; out[:-1, :] |= sk[1:, :]; out[:-1, :-1] |= sk[1:, 1:]
        out[1:, :-1] |= sk[:-1, 1:]; out[:-1, 1:] |= sk[1:, :-1]
    return Image.fromarray((out * 255).astype(np.uint8))


def elastic(im, rng):
    """Smooth random warp: the image is cut into ELASTIC_GRID x ELASTIC_GRID cells whose
    corners are moved randomly (border fixed); PIL maps every cell (MESH transform)."""
    W, H = im.size
    G = ELASTIC_GRID
    xs = np.round(np.linspace(0, W, G + 1)).astype(int)
    ys = np.round(np.linspace(0, H, G + 1)).astype(int)
    amp = ELASTIC_AMP * max(W, H)
    src = np.stack(np.meshgrid(xs, ys), -1).astype(float)    # src[i, j] = (x_j, y_i)
    src[1:-1, 1:-1] += rng.uniform(-amp, amp, size=(G - 1, G - 1, 2))
    mesh = []
    for i in range(G):
        for j in range(G):
            box = (int(xs[j]), int(ys[i]), int(xs[j + 1]), int(ys[i + 1]))
            quad = [src[i, j], src[i + 1, j], src[i + 1, j + 1], src[i, j + 1]]  # NW SW SE NE
            mesh.append((box, tuple(float(v) for pt in quad for v in pt)))
    return im.transform(im.size, Image.Transform.MESH, mesh, resample=Image.Resampling.BILINEAR)


def slant(im, rng):
    """Horizontal shear by s in [-SLANT, SLANT] (canvas widened so nothing is cut)."""
    s = rng.uniform(-SLANT, SLANT)
    W, H = im.size
    W2 = W + int(np.ceil(abs(s) * H))
    return im.transform((W2, H), Image.Transform.AFFINE, (1, s, -max(s, 0) * H, 0, 1, 0),
                        resample=Image.Resampling.BICUBIC, fillcolor=0)


def augment(hi, rng):
    """hi: high-resolution glyph ('L', ink=255). Returns a 32x32 uint8 array.
    Order: stroke weight -> elastic -> slant -> rotate -> crop/scale/shift into 32x32
    -> pen redraw or binarize -> blur -> noise (each step random, most with a probability)."""
    im = stroke(hi, rng)
    if rng.random() < P_ELASTIC:
        im = elastic(im, rng)
    if rng.random() < P_SLANT:
        im = slant(im, rng)
    im = im.rotate(rng.uniform(-ROT_DEG, ROT_DEG), resample=Image.Resampling.BICUBIC,
                   expand=True, fillcolor=0)
    side = int(rng.integers(SIDE[0], SIDE[1] + 1))
    lim = min(MAX_SHIFT, (OUT - side) // 2)
    dx, dy = (int(v) for v in rng.integers(-lim, lim + 1, size=2))
    im = place(to_square(im), side, dx, dy)
    u = rng.random()
    if u < P_PEN:
        im = pen(im, rng)
    elif u < P_PEN + P_BINARIZE:
        im = im.point(lambda v: 255 if v >= 128 else 0)
    if rng.random() < P_BLUR:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(*BLUR_R)))
    x = np.asarray(im, dtype=np.float32)
    if rng.random() < P_NOISE:
        x = x + rng.normal(0, 255 * rng.uniform(*NOISE_SD), size=x.shape)
    return np.clip(x, 0, 255).round().astype(np.uint8)
