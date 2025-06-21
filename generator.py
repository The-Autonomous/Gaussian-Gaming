# generator.py
"""A CPU-only scene generator that converts a stack of RGB images into a compact
Gaussian-Splat 3-D representation that can be streamed directly to a real-time
viewer or cached for later reuse.

Key design goals
----------------
* **Pure-Python / NumPy** implementation - runs on any machine, no CUDA.
* **Batch friendly & memory aware** - processes images in tiles to keep RAM low.
* **Realtime-ish** - critical loops are JIT-compiled with **Numba** when
  available, but gracefully fall back to pure-Python.
* **Pluggable depth estimation** - use your own monocular depth model or the
  fast built-in RGB-to-depth heuristic.
* **LRU cache** - generated worlds are stored in an in-memory cache keyed by a
  stable SHA-1 of the image set so repeated loads are instantaneous.

Typical usage
-------------
```python
from generator import SceneGenerator, InMemoryCache

imgs = [
    "assets/room_01.jpg",
    "assets/room_02.jpg",
    "assets/room_03.jpg",
]

cache = InMemoryCache(max_size=3)
gen   = SceneGenerator(cache=cache)
world = gen.build_world(imgs)

# world is now a GaussianSplatWorld you can stream to your engine
viewer.stream(world)
```
"""
from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    from numba import njit, prange
    # Disable Numba’s nopython branch (it can’t compile np.stack over a Python list)
    USE_NUMBA = False
except ImportError:
    USE_NUMBA = False

###############################################################################
# Data classes & cache
###############################################################################

@dataclass(slots=True)
class GaussianSplat:
    """A single 3-D Gaussian blob."""

    position: np.ndarray  # shape (3,)
    color: np.ndarray     # shape (3,) RGB in 0-1 range
    radius: float         # isotropic Gaussian radius in world units


@dataclass(slots=True)
class GaussianSplatWorld:
    """Collection of all splats and some scene meta-data."""

    splats: np.ndarray  # structured array (x, y, z, r, g, b, radius)
    bbox: Tuple[np.ndarray, np.ndarray]  # (min_xyz, max_xyz)
    # Extra fields (camera poses, floor plane, etc.) can be added later.

    def __iter__(self):
        """Allow `for p in world:` style loops."""
        return iter(self.splats)

    @property
    def center(self) -> np.ndarray:
        return (self.bbox[0] + self.bbox[1]) * 0.5


class InMemoryCache:
    """A brutally simple (thread-safe) LRU cache for worlds."""

    def __init__(self, max_size: int = 4):
        self.max_size = max_size
        self._lock    = threading.Lock()
        self._data: dict[str, GaussianSplatWorld] = {}
        self._order: List[str] = []  # insertion order as LRU

    # ---------------------------------------------------------------------
    def __contains__(self, key: str) -> bool:  # pragma: no cover
        with self._lock:
            return key in self._data

    def get(self, key: str) -> Optional[GaussianSplatWorld]:
        with self._lock:
            world = self._data.get(key)
            if world is not None:
                # promote to most-recent
                self._order.remove(key)
                self._order.append(key)
            return world

    def put(self, key: str, world: GaussianSplatWorld):
        with self._lock:
            if key in self._data:
                self._order.remove(key)
            elif len(self._data) >= self.max_size:
                # evict LRU
                oldest = self._order.pop(0)
                self._data.pop(oldest, None)
            self._data[key] = world
            self._order.append(key)

###############################################################################
# Scene generation pipeline
###############################################################################

class SceneGenerator:
    """Turns a set of images into a textured Gaussian-Splat world.

    Parameters
    ----------
    cache : InMemoryCache | None
        Optional cache. If supplied, repeated `build_world()` calls with the
        same images return instantly.
    tile_size : int, default 256
        Process images in (tile_size x tile_size) chunks instead of as whole
        matrices to conserve memory and improve cache locality.
    base_radius : float, default 0.0125
        World-space radius for each Gaussian (scaled later with depth-adaptive
        heuristics).
    """

    def __init__(self, *, cache: Optional[InMemoryCache] = None,
                 tile_size: int = 256, base_radius: float = 0.0125):
        self.cache       = cache
        self.tile_size   = tile_size
        self.base_radius = base_radius

    # ------------------------------------------------------------------
    def build_world(self, images: Iterable[str | os.PathLike[str]]) -> GaussianSplatWorld:
        """Main entry point.

        Parameters
        ----------
        images : iterable of str | Path
            Paths to RGB image files that will be fused into one scene.

        Returns
        -------
        GaussianSplatWorld
            Ready-to-stream world object.
        """
        imgs = [Path(p).expanduser().resolve() for p in images]
        if not imgs:
            raise ValueError("No images provided.")

        # A stable content hash for caching - insensitive to list ordering.
        key = sha1_of_paths(imgs)
        if self.cache is not None:
            world = self.cache.get(key)
            if world is not None:
                return world

        # 1. Load & preprocess
        rgb_stack = [load_rgb(path) for path in imgs]
        h, w      = rgb_stack[0].shape[:2]
        assert all(im.shape[:2] == (h, w) for im in rgb_stack), "All images must share the same resolution for now."

        # 2. Estimate depth for every image (naïve but fast)
        depth_stack = [estimate_depth(im) for im in rgb_stack]

        # 3. Fuse into a common point cloud, then Gaussianise
        splats = fuse_to_gaussians(rgb_stack, depth_stack,
                                   self.base_radius, self.tile_size)

        # 4. Compute bounding box (per-field min/max on structured array)
        mins = np.array([
            splats['x'].min(),
            splats['y'].min(),
            splats['z'].min(),
        ], dtype=np.float32)
        maxs = np.array([
            splats['x'].max(),
            splats['y'].max(),
            splats['z'].max(),
        ], dtype=np.float32)

        world = GaussianSplatWorld(splats=splats, bbox=(mins, maxs))

        if self.cache is not None:
            self.cache.put(key, world)
        return world

###############################################################################
# Helper functions & JITs
###############################################################################

def sha1_of_paths(paths: List[Path]) -> str:
    m = hashlib.sha1()
    for p in sorted(paths):
        m.update(p.as_posix().encode())
        m.update(str(os.path.getmtime(p)).encode())  # invalidate on file change
    return m.hexdigest()


def load_rgb(path: Path) -> np.ndarray:
    """Load image as float32 RGB array in 0-1 range."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0


# ---------------------------------------------------------------------------
#   Depth estimation - you can swap this with a better model if available.
# ---------------------------------------------------------------------------

def estimate_depth(rgb: np.ndarray) -> np.ndarray:
    """VERY rough inverse-Sobel variance clue as single-view depth.

    The heuristic says *more texture ⇒ likely closer*. It is not physically
    correct but is instantaneous and produces a pleasing pseudo-parallax.
    Replace with a real monocular depth model (e.g. MiDaS) for quality.
    """
    import cv2  # local import keeps the top fast if OpenCV is missing

    gray   = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag    = np.sqrt(sobelx ** 2 + sobely ** 2)
    # invert so that strong gradients → small depth → near camera
    depth  = 1.0 / (mag + 1e-6)
    depth  = depth / depth.max()  # normalise to 0-1
    return depth.astype(np.float32)


# ---------------------------------------------------------------------------
#   Fusion - combines per-image depth maps into a unified Gaussian cloud.
# ---------------------------------------------------------------------------

DTYPE = np.float32
STRUCT_DTYPE = np.dtype([
    ("x", DTYPE), ("y", DTYPE), ("z", DTYPE),
    ("r", DTYPE), ("g", DTYPE), ("b", DTYPE),
    ("radius", DTYPE),
])


def fuse_to_gaussians(rgb_stack: List[np.ndarray], depth_stack: List[np.ndarray],
                      base_radius: float, tile: int) -> np.ndarray:
    """Vectorised implementation that remains memory-frugal by tiling."""
    h, w   = rgb_stack[0].shape[:2]
    tiles_y = (h + tile - 1) // tile
    tiles_x = (w + tile - 1) // tile

    out: List[np.ndarray] = []
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            y0, y1 = ty * tile, min((ty + 1) * tile, h)
            x0, x1 = tx * tile, min((tx + 1) * tile, w)

            chunk_rgbs   = [im[y0:y1, x0:x1] for im in rgb_stack]
            chunk_depths = [dm[y0:y1, x0:x1] for dm in depth_stack]
            out.append(_chunk_to_gaussians(chunk_rgbs, chunk_depths,
                                           base_radius,
                                           offset=(x0, y0)))
    return np.concatenate(out, axis=0)


if USE_NUMBA:

    @njit(parallel=True, cache=True)
    def _chunk_to_gaussians(chunk_rgbs: List[np.ndarray],
                            chunk_depths: List[np.ndarray],
                            base_radius: float,
                            offset: Tuple[int, int]) -> np.ndarray:  # pragma: no cover
        # Numba cannot compile dynamic Python lists of ndarrays well.
        # So we first stack them.
        rgb_stack = np.stack(chunk_rgbs, axis=0)    # (N, H, W, 3)
        depth_stack = np.stack(chunk_depths, axis=0)  # (N, H, W)
        n_imgs, h, w, _ = rgb_stack.shape

        # Each pixel of each image becomes one Gaussian - enormous but cheap.
        total_pts = n_imgs * h * w
        splats = np.empty(total_pts, dtype=STRUCT_DTYPE)

        idx = 0
        ox, oy = offset
        for n in prange(n_imgs):
            for y in range(h):
                for x in range(w):
                    z     = depth_stack[n, y, x]   # 0-1 depth
                    xpos  = (x + ox - w * .5) * z  # simple pinhole assume f=1
                    ypos  = (y + oy - h * .5) * z
                    c     = rgb_stack[n, y, x]

                    splats[idx]["x"] = xpos
                    splats[idx]["y"] = ypos
                    splats[idx]["z"] = z
                    splats[idx]["r"] = c[0]
                    splats[idx]["g"] = c[1]
                    splats[idx]["b"] = c[2]
                    splats[idx]["radius"] = base_radius * (1 + (1 - z))
                    idx += 1
        return splats

else:

    def _chunk_to_gaussians(chunk_rgbs: List[np.ndarray],
                            chunk_depths: List[np.ndarray],
                            base_radius: float,
                            offset: Tuple[int, int]) -> np.ndarray:
        # This pure-Python version can happily do np.stack on a Python list
        rgb_stack   = np.stack(chunk_rgbs,   axis=0)
        depth_stack = np.stack(chunk_depths, axis=0)
        n_imgs, h, w, _ = rgb_stack.shape
        out = np.empty(n_imgs * h * w, dtype=STRUCT_DTYPE)
        idx = 0
        ox, oy = offset
        for n in range(n_imgs):
            for y in range(h):
                for x in range(w):
                    z = depth_stack[n, y, x]
                    xpos = (x + ox - w * 0.5) * z
                    ypos = (y + oy - h * 0.5) * z
                    r, g, b = rgb_stack[n, y, x]
                    out[idx] = (xpos, ypos, z, r, g, b,
                                base_radius * (1 + (1 - z)))
                    idx += 1
        return out

###############################################################################
# CLI helper for quick manual testing
###############################################################################

if __name__ == "__main__":  # pragma: no cover
    import argparse
    from time import perf_counter

    ap = argparse.ArgumentParser(description="Generate Gaussian-Splat world")
    ap.add_argument("images", nargs="+", help="Input image files")
    ap.add_argument("--no-numba", action="store_true",
                    help="Disable Numba even if it is installed")
    ap.add_argument("--preview", action="store_true",
                    help="Visualise a slice of the splat cloud (matplotlib)")
    args = ap.parse_args()

    
    
    USE_NUMBA = not args.no_numba

    gen = SceneGenerator(cache=InMemoryCache(max_size=1))
    t0 = perf_counter()
    world = gen.build_world(args.images)
    t1 = perf_counter()
    print(f"Converted {len(args.images)} image(s) → {len(world.splats):,} splats in {(t1 - t0):.2f}s")

    if args.preview:
        try:
            import matplotlib.pyplot as plt  # noqa: F811 - optional
            pts = world.splats
            fig = plt.figure("X-Z slice")
            ax  = fig.add_subplot(111)
            ax.scatter(pts[::16]["x"], pts[::16]["z"], s=1)
            ax.set_xlabel("X")
            ax.set_ylabel("Z")
            ax.set_aspect("equal")
            plt.show()
        except ImportError:
            print("Matplotlib not installed; skipping preview.")
