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
import sys

from generator import SceneGenerator, InMemoryCache
from graphics import Viewer
import pyglet

from utils import (
    scan_images,
    find_min_resolution,
    uniformise_images,
    world_to_scene,
)




def main() -> None:
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "images").expanduser()
    if not folder.is_dir():
        sys.exit(f"{folder} is not a directory")

    imgs = scan_images(folder)
    if not imgs:
        sys.exit(f"No images found in {folder}")

    # 1. Harmonise resolutions
    min_w, min_h = find_min_resolution(imgs)
    uniform_imgs, (min_w, min_h) = uniformise_images(imgs, (min_w, min_h))

    # 2. Build world
    gen = SceneGenerator(cache=InMemoryCache(max_size=1))
    world = gen.build_world(uniform_imgs)

    # 3. View
    scene = world_to_scene(world)
    # normalize pixel-space X/Y into unit range so they fit the frustum
    scene.positions[:, 0] /= min_w
    scene.positions[:, 1] /= min_h
    Viewer(scene)
    pyglet.app.run()


if __name__ == "__main__":
    main()
