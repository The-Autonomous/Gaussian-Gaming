from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import tempfile

import numpy as np
from PIL import Image


def scan_images(folder: Path) -> List[Path]:
    """Return paths of common image files in *folder*."""
    exts = {".png", ".jpg", ".jpeg"}
    return [p for p in folder.iterdir() if p.suffix.lower() in exts]


def find_min_resolution(paths: List[Path]) -> Tuple[int, int]:
    """Smallest width/height across *paths*."""
    widths, heights = [], []
    for p in paths:
        with Image.open(p) as im:
            w, h = im.size
            widths.append(w)
            heights.append(h)
    return min(widths), min(heights)


def uniformise_images(
    paths: List[Path], dims: Tuple[int, int], max_size: int = 1024
) -> Tuple[List[Path], Tuple[int, int]]:
    """Centre-crop all images to the same size and optionally downscale.

    Parameters
    ----------
    paths : list of Path
        Input image files.
    dims : (int, int)
        Target dimensions before optional downscale.
    max_size : int, default 1024
        Maximum allowed width or height. Images larger than this will be
        resized after cropping to keep the total point count reasonable.
    """
    tw, th = dims
    scale = 1.0
    if max(tw, th) > max_size:
        scale = max_size / float(max(tw, th))
    final_dims = (int(tw * scale), int(th * scale))

    out_paths = []
    tmpdir = Path(tempfile.mkdtemp(prefix="uniform_"))
    for idx, src in enumerate(paths):
        with Image.open(src) as im:
            w, h = im.size
            if (w, h) != (tw, th):
                left = (w - tw) // 2
                top = (h - th) // 2
                right = left + tw
                bottom = top + th
                im = im.crop((left, top, right, bottom))
            if scale != 1.0:
                im = im.resize(final_dims, Image.LANCZOS)
            if (w, h) == final_dims and scale == 1.0:
                out_paths.append(src)
            else:
                dst = tmpdir / f"uniform_{idx:04d}{src.suffix.lower()}"
                im.save(dst, quality=95)
                out_paths.append(dst)
    return out_paths, final_dims


def world_to_scene(world):
    """Convert a GaussianSplatWorld to the lightweight object expected by Viewer."""
    splats = world.splats

    class Scene:
        pass

    scn = Scene()
    scn.positions = np.stack((splats["x"], splats["y"], splats["z"]), axis=1).astype(np.float32)
    scn.colors = np.column_stack(
        (splats["r"], splats["g"], splats["b"], np.ones(len(splats)))
    ).astype(np.float32)
    scn.sizes = (splats["radius"] * 1000.0).astype(np.float32)
    scn.center = world.center.astype(np.float32)
    return scn

