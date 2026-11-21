"""get_fonts.py: step 1, put the 18 math font files into fonts/.

PIPELINE  data_collection/: 1 get_fonts -> 2 make_dataset -> 3 data_figures -> 4 hasy
          training/:        5 tune (optional) -> 6 train
          evaluation/:      7 final_test (once, at the end)
THIS FILE step 1
READS     fonts_config.FONTS; a local TeX install if any; CTAN (internet)
WRITES    fonts/*.otf
USED BY   make_dataset (reads fonts/)
RUN       python src/data_collection/get_fonts.py
------------------------------------------------------------------------------

Collect the math fonts listed in fonts_config.py into fonts/.

Order of attempts for each font:
  1. already in fonts/            -> skip
  2. a local TeX install          (TeX Live / MiKTeX / Colab texlive) -> copy
  3. download from CTAN, trying several mirrors, 3 tries each:
     the single .otf file first, then the package zip

Usage:  python src/data_collection/get_fonts.py [--fonts-dir fonts]
Prints a table of what was found; fonts that fail are simply left out of the
dataset (make_dataset.py only uses fonts present in fonts/).
"""
import argparse, io, os, re, shutil, subprocess, sys, time, urllib.request, zipfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from fonts_config import FONTS

# mirrors.ctan.org redirects to a random mirror, which occasionally fails or is
# out of sync, so a few fixed mirrors are tried after it.
MIRRORS = ["https://mirrors.ctan.org/",
           "https://mirror.its.dal.ca/ctan/",
           "https://ctan.math.illinois.edu/",
           "https://ftp.math.utah.edu/pub/ctan/tex-archive/",
           "https://ctan.net/"]
TRIES = 3


def local_tex_dirs():
    """Folders where an installed TeX distribution or the OS keeps OpenType fonts
    (only the ones that exist on this machine)."""
    dirs = []
    try:  # TeX Live / MiKTeX via kpsewhich
        for var in ["TEXMFDIST", "TEXMFMAIN", "TEXMFLOCAL"]:
            out = subprocess.run(["kpsewhich", f"-var-value={var}"], capture_output=True,
                                 text=True, timeout=10).stdout.strip()
            if out:
                dirs.append(Path(out) / "fonts" / "opentype")
    except Exception:
        pass
    home = Path.home()
    dirs += [Path("/usr/share/texlive/texmf-dist/fonts/opentype"),
             Path("/usr/share/texmf/fonts/opentype"),
             Path("/usr/share/fonts"),
             home / "AppData/Local/Programs/MiKTeX/fonts/opentype",
             Path("C:/Program Files/MiKTeX/fonts/opentype"),
             home / "AppData/Local/Microsoft/Windows/Fonts",
             Path("C:/Windows/Fonts")]
    dirs += [p / "texmf-dist/fonts/opentype" for p in Path("C:/texlive").glob("20*")] \
        if Path("C:/texlive").exists() else []
    return [d for d in dirs if d.exists()]


def find_local(regex, dirs):
    """First file under `dirs` whose name matches `regex` (case-insensitive), else None."""
    pat = re.compile(regex, re.I)
    for d in dirs:
        for root, _, files in os.walk(d):
            for f in files:
                if pat.search(f):
                    return Path(root) / f
    return None


def fetch(url):
    """Download `url` and return its bytes (2 min timeout)."""
    req = urllib.request.Request(url, headers={"User-Agent": "aps360-fonts"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def is_font(data):
    """True if the bytes start with an OpenType/TrueType signature (not an HTML error page)."""
    return data[:4] in (b"OTTO", b"\x00\x01\x00\x00", b"true")


_zip_cache = {}
def from_ctan(pkg, regex, inner):
    """Returns (file name, bytes). Raises with every error seen if all attempts fail."""
    errors = []
    for mirror in MIRRORS:
        for t in range(TRIES):
            try:                                   # 1) the .otf file itself
                data = fetch(mirror + pkg + "/" + inner)
                if is_font(data):
                    return Path(inner).name, data
                errors.append(f"{mirror}: not a font file")
                break
            except Exception as e:
                errors.append(f"{mirror} file try {t + 1}: {type(e).__name__}: {e}")
                time.sleep(1 + t)
        for t in range(TRIES):
            try:                                   # 2) the whole package zip
                key = mirror + pkg
                if key not in _zip_cache:
                    _zip_cache[key] = zipfile.ZipFile(io.BytesIO(fetch(mirror + pkg + ".zip")))
                z = _zip_cache[key]
                names = [n for n in z.namelist() if re.search(regex, n, re.I)]
                if names:
                    return Path(names[0]).name, z.read(names[0])
                errors.append(f"{mirror}: no file matching {regex} in {pkg}.zip")
                break
            except Exception as e:
                errors.append(f"{mirror} zip try {t + 1}: {type(e).__name__}: {e}")
                time.sleep(1 + t)
    raise RuntimeError("; ".join(errors[-4:]))


def main():
    """For each font: keep it if already in fonts/, else copy it from a local TeX
    install, else download it from CTAN; then print a status table."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonts-dir", default="fonts")
    ap.add_argument("--no-download", action="store_true")
    a = ap.parse_args()
    out = Path(a.fonts_dir); out.mkdir(parents=True, exist_ok=True)
    dirs = local_tex_dirs()
    pat_any = lambda rx: re.compile(rx, re.I)
    status = {}
    for name, (pkg, rx, split, _, inner) in FONTS.items():
        have = [p for p in out.iterdir() if pat_any(rx).search(p.name)]
        if have:
            status[name] = ("present", have[0].name); continue
        p = find_local(rx, dirs)
        if p:
            shutil.copy(p, out / p.name); status[name] = ("local TeX", p.name); continue
        if a.no_download:
            status[name] = ("MISSING", ""); continue
        try:
            fname, data = from_ctan(pkg, rx, inner)
            (out / Path(fname).name).write_bytes(data)
            status[name] = ("CTAN", Path(fname).name)
        except Exception as e:
            status[name] = ("MISSING", str(e))
    print(f"{'font':12s} {'split':6s} {'source':10s} file")
    for name, (src, info) in status.items():
        print(f"{name:12s} {FONTS[name][2]:6s} {src:10s} {info}")
    n_ok = sum(s != "MISSING" for s, _ in status.values())
    print(f"\n{n_ok}/{len(FONTS)} fonts available in {out}/")
    if n_ok < len(FONTS):
        print("Missing fonts: download the .otf by hand into fonts/ from")
        for name, (src, _) in status.items():
            if src == "MISSING":
                pkg, _, _, _, inner = FONTS[name]
                print(f"  {name}: {MIRRORS[0]}{pkg}/{inner}")


if __name__ == "__main__":
    main()
