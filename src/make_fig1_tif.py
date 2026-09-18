"""
make_fig1_tif.py — Derive the submission TIFF of Figure 1 from its vector PDF.

Why: Figure 1 ("Composition of the public FDA shortage list") was assembled by
the original build session and ships as Figure_1.png + Figure_1.pdf only; no
generator script and no TIFF existed in the package. Rather than inventing a
re-render, we rasterize the shipped vector PDF at 300 dpi and write an LZW
TIFF, so the TIFF is pixel-faithful to the delivered figure by construction.

Output: figures_public/Figure_1.tif (RGB, LZW, 300 dpi tagged).
"""
from pathlib import Path

import pymupdf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT.parent                     # package root (18_2026.9.16_...)
SRC = PKG / "02_Figures" / "Figure_1.pdf"   # shipped vector source
OUT = ROOT / "figures_public" / "Figure_1.tif"
if not OUT.parent.exists():          # shim-safe guard (exist_ok is not honoured)
    OUT.parent.mkdir(parents=True, exist_ok=True)

DPI = 300
doc = pymupdf.open(SRC)
page = doc[0]
pix = page.get_pixmap(dpi=DPI, alpha=False)
im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
# tag physical DPI so the canvas prints at 300 ppi
im.save(OUT, compression="tiff_lzw", dpi=(DPI, DPI))
print(f"Figure_1.tif: {im.size[0]}x{im.size[1]} px, dpi={DPI}, LZW, mode={im.mode}")
