#!/usr/bin/env python3
# main.py   -   scan a folder, force all images to the same size, build & show
#
# Usage
#   python main.py                 # scans ./images
#   python main.py myfolder/       # scans custom folder
#
# Requirements
#   pip install pillow numpy pyglet pynput

from pathlib import Path
import sys, tempfile
from typing import List, Tuple

import numpy as np
from PIL import Image

from generator import SceneGenerator, InMemoryCache
from graphics import Viewer
import pyglet


def scan_images(folder: Path) -> List[Path]:
    """Scan a folder for image files (png, jpg, jpeg) and return their paths."""
    exts = {".png", ".jpg", ".jpeg"}
    return [p for p in folder.iterdir() if p.suffix.lower() in exts]


def find_min_resolution(paths: List[Path]) -> Tuple[int, int]:
    """Find minimum width and height across all images."""
    widths, heights = [], []
    for p in paths:
        with Image.open(p) as im:
            w, h = im.size
            widths.append(w)
            heights.append(h)
    return min(widths), min(heights)


def uniformise_images(paths: List[Path], dims: Tuple[int, int]) -> List[Path]:
    """Centre-crop all images to the same dimensions (tw, th)."""
    tw, th = dims
    out_paths = []
    tmpdir = Path(tempfile.mkdtemp(prefix="uniform_"))
    for idx, src in enumerate(paths):
        with Image.open(src) as im:
            w, h = im.size
            if (w, h) == (tw, th):
                out_paths.append(src)  # already correct size
                continue

            # Centre-crop to smallest common rectangle
            left = (w - tw) // 2
            top = (h - th) // 2
            right = left + tw
            bottom = top + th
            cropped = im.crop((left, top, right, bottom))
            # If you prefer a down-scale instead of a crop:
            # cropped = im.resize((tw, th), Image.NEAREST)
            dst = tmpdir / f"uniform_{idx:04d}{src.suffix.lower()}"
            cropped.save(dst, quality=95)
            out_paths.append(dst)

    return out_paths


def world_to_scene(world, scale_xy: tuple[float, float] = (1.0, 1.0)):
    """Convert ``GaussianSplatWorld`` to the lightweight object expected by ``Viewer``.

    Parameters
    ----------
    world : GaussianSplatWorld
        Input world generated by :class:`SceneGenerator`.
    scale_xy : tuple of float, optional
        Scale factors for the X and Y axes. This is useful when the caller
        normalises the splats to a specific range (for example by the source
        image resolution).
    """
    splats = world.splats
    scale_x, scale_y = scale_xy

    class Scene:
        pass

    scn = Scene()
    scn.positions = np.stack(
        (splats["x"] / scale_x, splats["y"] / scale_y, splats["z"]),
        axis=1,
    ).astype(np.float32)
    scn.colors = np.column_stack(
        (splats["r"], splats["g"], splats["b"], np.ones(len(splats)))
    ).astype(np.float32)
    scn.sizes = (splats["radius"] * 1000.0).astype(np.float32)
    scn.center = np.array(
        [world.center[0] / scale_x, world.center[1] / scale_y, world.center[2]],
        dtype=np.float32,
    )
    return scn


def main() -> None:
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "images").expanduser()
    if not folder.is_dir():
        sys.exit(f"{folder} is not a directory")

    imgs = scan_images(folder)
    if not imgs:
        sys.exit(f"No images found in {folder}")

    # 1. Harmonise resolutions
    min_w, min_h = find_min_resolution(imgs)
    uniform_imgs = uniformise_images(imgs, (min_w, min_h))

    # 2. Build world
    gen = SceneGenerator(cache=InMemoryCache(max_size=1))
    world = gen.build_world(uniform_imgs)

    # 3. View
    scene = world_to_scene(world, scale_xy=(min_w, min_h))
    Viewer(scene)
    pyglet.app.run()


if __name__ == "__main__":
    main()
