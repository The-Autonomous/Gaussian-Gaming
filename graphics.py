# graphics.py
"""Gaussian-splat real-time viewer (CPU-prep, GPU-render)

* **Controls**: W-A-S-D to move on the X-Z plane, SPACE / LEFT-SHIFT for up/down,
  mouse-look to rotate the camera, ESC to release pointer, Q to quit.
* Requires **pyglet ≥ 2.0**, **numpy**, **pynput**, and the accompanying
  **generator.py** supplying a `Scene` object with XYZ positions, RGBA colours and
  per-point sizes (in pixels).
* Stays CPU-only for the heavy lifting; we use modern OpenGL via pyglet to draw
  sprites (one point = one Gaussian splat).

Run directly:
```bash
python graphics.py image1.jpg image2.jpg ...
```
If no images are given, a procedural test cloud is generated.
"""
from __future__ import annotations

import math, sys, threading, ctypes
from ctypes import POINTER, byref
from pathlib import Path
from typing import Sequence, TypeAlias

import numpy as np
import pyglet
from pyglet import gl
from pyglet.window import key

from pynput import keyboard as kb_listener, mouse as mouse_listener

# ----------------------------------------------------------------------------
# Typing helpers
# ----------------------------------------------------------------------------
_VECTOR: TypeAlias = Sequence[float]  # 3-component vector (x, y, z)

# ----------------------------------------------------------------------------
# Data → GPU helpers
# ----------------------------------------------------------------------------

def _create_shader_program() -> gl.GLuint:
    """Compiles a minimalist Gaussian-sprite shader and returns its program id."""
    vertex_src = (b"""
    #version 330 core
    layout(location = 0) in vec3 in_pos;
    layout(location = 1) in vec4 in_col;
    layout(location = 2) in float in_size;
    uniform mat4 u_mvp;
    out vec4 v_col;
    void main() {
        v_col = in_col;
        gl_PointSize = in_size;
        gl_Position = u_mvp * vec4(in_pos, 1.0);
    }
    """
    )
    fragment_src = (b"""
    #version 330 core
    in vec4 v_col;
    out vec4 fragColor;
    void main() {
        float d = length(gl_PointCoord - vec2(0.5));
        float alpha = exp(-32.0 * d * d);  // sharp gaussian fall-off
        fragColor = vec4(v_col.rgb, v_col.a * alpha);
    }
    """
    )

    def _compile(src: bytes, shader_type: int) -> gl.GLuint:
        shader = gl.glCreateShader(shader_type)

        # Build a pointer-to-pointer (const char **)
        src_buf  = ctypes.c_char_p(src)               # null-terminated
        src_ptrs = ctypes.cast(
            ctypes.pointer(src_buf),                  # char **
            POINTER(POINTER(gl.GLchar))
        )

        length = gl.GLint(len(src))
        gl.glShaderSource(shader, 1, src_ptrs, byref(length))
        gl.glCompileShader(shader)

        # Check compilation status
        status = gl.GLint()
        gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS, byref(status))
        if not status.value:
            log_len = gl.GLint()
            gl.glGetShaderiv(shader, gl.GL_INFO_LOG_LENGTH, byref(log_len))
            buf = (gl.GLchar * log_len.value)()
            gl.glGetShaderInfoLog(shader, log_len, None, buf)
            raise RuntimeError(buf.value.decode("utf-8", "replace"))
        return shader

    prog = gl.glCreateProgram()
    vs = _compile(vertex_src, gl.GL_VERTEX_SHADER)
    fs = _compile(fragment_src, gl.GL_FRAGMENT_SHADER)
    gl.glAttachShader(prog, vs)
    gl.glAttachShader(prog, fs)
    gl.glLinkProgram(prog)
    # Delete shaders once linked
    gl.glDeleteShader(vs)
    gl.glDeleteShader(fs)
    return prog


def _perspective(fov_y: float, aspect: float, z_near: float, z_far: float) -> np.ndarray:
    """Generates a perspective projection matrix."""
    f = 1.0 / math.tan(math.radians(fov_y) / 2)
    range_inv = 1.0 / (z_near - z_far)
    return np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (z_near + z_far) * range_inv, -1],
        [0, 0, z_near * z_far * range_inv * 2, 0],
    ], dtype=np.float32)


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Computes a right-handed look-at matrix."""
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.identity(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    trans = np.identity(4, dtype=np.float32)
    trans[:3, 3] = -eye
    return m @ trans

# ----------------------------------------------------------------------------
# Camera
# ----------------------------------------------------------------------------
class Camera:
    """Simple FPS-style camera with yaw/pitch control."""

    def __init__(self, pos: _VECTOR = (0.0, 0.0, 3.0)) -> None:
        self.position = np.array(pos, dtype=np.float32)
        self.yaw: float = 0.0  # degrees
        self.pitch: float = 0.0  # degrees
        self._move_dir = np.zeros(3, dtype=np.float32)
        self.move_speed = 4.0  # units/s
        self.sprint_mult = 2.5
        self.mouse_sens = 0.15  # degs per px

    # ------------------------------------------------------------------
    # Movement helpers
    # ------------------------------------------------------------------
    def forward(self) -> np.ndarray:
        yaw_rad = math.radians(self.yaw)
        return np.array([
            math.sin(yaw_rad),
            0,
            -math.cos(yaw_rad),
        ], dtype=np.float32)

    def right(self) -> np.ndarray:
        yaw_rad = math.radians(self.yaw)
        return np.array([
            math.cos(yaw_rad),
            0,
            math.sin(yaw_rad),
        ], dtype=np.float32)

    def up(self) -> np.ndarray:
        return np.array([0, 1, 0], dtype=np.float32)

    # ------------------------------------------------------------------
    # External interface
    # ------------------------------------------------------------------
    def step(self, dt: float, sprint: bool) -> None:
        velocity = self.move_speed * (self.sprint_mult if sprint else 1.0)
        self.position += self._move_dir * velocity * dt

    def set_motion(self, fwd: float, right: float, up: float) -> None:
        self._move_dir = (
            self.forward() * fwd + self.right() * right + self.up() * up
        )

    def rotate(self, dx: float, dy: float) -> None:
        self.yaw += dx * self.mouse_sens
        self.pitch = max(-89.9, min(89.9, self.pitch + dy * self.mouse_sens))

    def view_matrix(self) -> np.ndarray:
        yaw_rad = math.radians(self.yaw)
        pitch_rad = math.radians(self.pitch)
        dir_vec = np.array(
            [
                math.cos(pitch_rad) * math.sin(yaw_rad),
                math.sin(pitch_rad),
                -math.cos(pitch_rad) * math.cos(yaw_rad),
            ],
            dtype=np.float32,
        )
        return _look_at(self.position, self.position + dir_vec, self.up())

# ----------------------------------------------------------------------------
# Listener threads (pynput) - decoupled from pyglet main loop
# ----------------------------------------------------------------------------
class _InputState:
    def __init__(self):
        self.keys = set()
        self.dx = 0.0
        self.dy = 0.0
        self.sprint = False
        self._lock = threading.Lock()

    # Keyboard callbacks
    def on_press(self, key_evt):
        with self._lock:
            if isinstance(key_evt, kb_listener.KeyCode):
                self.keys.add(key_evt.char.lower())
            elif key_evt == kb_listener.Key.shift:
                self.sprint = True
            elif key_evt == kb_listener.Key.esc:
                pyglet.app.exit()

    def on_release(self, key_evt):
        with self._lock:
            if isinstance(key_evt, kb_listener.KeyCode):
                self.keys.discard(key_evt.char.lower())
            elif key_evt == kb_listener.Key.shift:
                self.sprint = False

    # Mouse move callback
    def on_move(self, x, y):
        # Captured via pynput; we accumulate deltas each frame
        with self._lock:
            # We only track deltas; reset will happen in main loop
            # (pynput gives absolute pos; pyglet can provide deltas but we keep
            # everything in this single unified input state class.)
            pass  # For simplicity we rely on pyglet's relative motion below

    # Safe snapshot called from render thread
    def snapshot(self):
        with self._lock:
            return self.keys.copy(), self.sprint

_input_state = _InputState()

# Start keyboard listener (runs in its own thread; non-blocking)
kb_listener.Listener(
    on_press=_input_state.on_press,
    on_release=_input_state.on_release,
    suppress=False,
).start()

# ----------------------------------------------------------------------------
# Viewer window
# ----------------------------------------------------------------------------
class Viewer(pyglet.window.Window):
    """A pyglet window displaying a Gaussian‐splat scene."""

    def __init__(self, scene, width: int = 1280, height: int = 720):
        super().__init__(width=width, height=height, caption="Gaussian Viewer", resizable=True)
        self.set_exclusive_mouse(True)  # capture cursor
        self.scene = scene
        self.cam = Camera(pos=scene.center if hasattr(scene, "center") else (0, 0, 3))
        self.shader = _create_shader_program()
        self.mvp_loc = gl.glGetUniformLocation(self.shader, b"u_mvp")
        self._setup_buffers(scene)
        pyglet.clock.schedule(self._update)

    # ------------------------------------------------------------------
    # GL buffers
    # ------------------------------------------------------------------
    def _setup_buffers(self, scene):
        pts   = scene.positions.astype(np.float32)
        cols  = scene.colors.astype(np.float32)
        sizes = scene.sizes.astype(np.float32)
        self.count = len(pts)

        # ---------- 1. raw VBO uploads ----------
        self.vbo_pos  = gl.GLuint(0)
        self.vbo_col  = gl.GLuint(0)
        self.vbo_size = gl.GLuint(0)

        gl.glGenBuffers(1, ctypes.byref(self.vbo_pos))
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_pos)
        gl.glBufferData(gl.GL_ARRAY_BUFFER,
                        pts.nbytes,
                        pts.ctypes.data_as(ctypes.POINTER(gl.GLfloat)),
                        gl.GL_STATIC_DRAW)

        gl.glGenBuffers(1, ctypes.byref(self.vbo_col))
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_col)
        gl.glBufferData(gl.GL_ARRAY_BUFFER,
                        cols.nbytes,
                        cols.ctypes.data_as(ctypes.POINTER(gl.GLfloat)),
                        gl.GL_STATIC_DRAW)

        gl.glGenBuffers(1, ctypes.byref(self.vbo_size))
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_size)
        gl.glBufferData(gl.GL_ARRAY_BUFFER,
                        sizes.nbytes,
                        sizes.ctypes.data_as(ctypes.POINTER(gl.GLfloat)),
                        gl.GL_STATIC_DRAW)

        # ---------- 2. ONE VAO THAT REMEMBERS THE ABOVE ----------
        self.vao = gl.GLuint(0)
        gl.glGenVertexArrays(1, ctypes.byref(self.vao))
        gl.glBindVertexArray(self.vao)

        # positions  (location = 0)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_pos)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0,
                                ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(0)

        # colours    (location = 1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_col)
        gl.glVertexAttribPointer(1, 4, gl.GL_FLOAT, gl.GL_FALSE, 0,
                                ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(1)

        # sizes      (location = 2)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_size)
        gl.glVertexAttribPointer(2, 1, gl.GL_FLOAT, gl.GL_FALSE, 0,
                                ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(2)

        # tidy
        gl.glBindVertexArray(0)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def _update(self, dt):
        keys, sprint = _input_state.snapshot()
        fwd = ("w" in keys) - ("s" in keys)
        strafe = ("d" in keys) - ("a" in keys)
        lift = ("space" in keys) - ("c" in keys)
        self.cam.set_motion(float(fwd), float(strafe), float(lift))
        self.cam.step(dt, sprint)

    # Mouse motion from pyglet (relative)
    def on_mouse_motion(self, x, y, dx, dy):
        self.cam.rotate(dx, dy)

    def on_draw(self):
        self.clear()

        # MVP upload …
        gl.glUseProgram(self.shader)
        aspect = self.width / self.height if self.height else 1.0
        proj   = _perspective(70.0, aspect, 0.1, 1000.0)
        mvp    = proj @ self.cam.view_matrix()
        gl.glUniformMatrix4fv(self.mvp_loc, 1, gl.GL_TRUE,
                            mvp.ctypes.data_as(ctypes.POINTER(gl.GLfloat)))

        # bind VAO recorded earlier – no per-frame attribute calls needed
        gl.glBindVertexArray(self.vao)

        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)

        gl.glDrawArrays(gl.GL_POINTS, 0, self.count)

        # tidy
        gl.glBindVertexArray(0)
        gl.glUseProgram(0)

    def on_key_press(self, symbol, modifiers):
        if symbol == key.ESCAPE:
            self.set_exclusive_mouse(False)
        elif symbol == key.Q:
            pyglet.app.exit()

# ----------------------------------------------------------------------------
# Scene loader helpers (relies on generator.py)
# ----------------------------------------------------------------------------

def _load_scene_from_images(img_paths):
    """Utility used by the demo CLI to build a scene from images."""
    from generator import SceneGenerator, InMemoryCache
    from utils import find_min_resolution, uniformise_images, world_to_scene

    min_w, min_h = find_min_resolution(img_paths)
    imgs, (min_w, min_h) = uniformise_images(img_paths, (min_w, min_h))
    world = SceneGenerator(cache=InMemoryCache(max_size=1)).build_world(imgs)
    scene = world_to_scene(world)
    scene.positions[:, 0] /= min_w
    scene.positions[:, 1] /= min_h
    return scene


def _create_test_scene(num_points: int = 10_000):
    class _Scene:
        def __init__(self, n):
            self.positions = np.random.uniform(-5, 5, (n, 3)).astype(np.float32)
            self.colors = np.random.uniform(0.2, 1.0, (n, 4)).astype(np.float32)
            self.colors[:, 3] = 1.0  # full alpha
            self.sizes = np.full((n,), 8, dtype=np.float32)
            self.center = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    return _Scene(num_points)

# ----------------------------------------------------------------------------
# __main__ entrypoint
# ----------------------------------------------------------------------------

def main(argv: list[str] | None = None):
    argv = argv if argv is not None else sys.argv
    imgs = [Path(p) for p in argv[1:] if Path(p).is_file()]
    if imgs:
        scene = _load_scene_from_images(imgs)
    else:
        print("[graphics] No images supplied - falling back to procedural test scene.")
        scene = _create_test_scene()

    Viewer(scene)
    pyglet.app.run()


if __name__ == "__main__":
    main()
