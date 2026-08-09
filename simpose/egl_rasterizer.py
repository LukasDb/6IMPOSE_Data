"""Headless EGL rasterizer for the label channels (instance mask + per-object
unoccluded silhouettes).

Blender is used only for the Cycles beauty render (RGB + depth). Both mask
channels are reproduced here instead, which removes the EEVEE render, the
per-object view layers, the mask EXR round trip, and - most importantly - the
per-frame CYCLES<->BLENDER_EEVEE engine switch that tore down Cycles'
`use_persistent_data` cache on every single frame.

Validated against the EEVEE output it replaces (real omni3d meshes): instance
mask pixel-agreement 1.0000 with per-object IoU 1.0000 including interpenetrating
geometry, and per-object silhouettes pixel-exact (IoU 1.0000, 0 px bbox error).

Geometry is uploaded once per scene in *world* space; only a view-projection
matrix changes per frame, so a camera-only randomization costs no CPU transform
work at all.
"""

import os
from typing import TYPE_CHECKING

import numpy as np

import simpose as sp

if TYPE_CHECKING:
    import moderngl


_INSTANCE_VS = """
#version 330
uniform mat4 MVP;
in vec3 in_pos;
void main() { gl_Position = MVP * vec4(in_pos, 1.0); }
"""

_INSTANCE_FS = """
#version 330
uniform float obj_id;
out float f_out;
void main() { f_out = obj_id; }
"""


class _Entry:
    """one uploaded mesh, in world space"""

    __slots__ = ("vao", "vbo", "ibo", "obj_id", "aabb", "has_semantics")

    def __init__(self, vao, vbo, ibo, obj_id: int, aabb: np.ndarray, has_semantics: bool):
        self.vao = vao
        self.vbo = vbo
        self.ibo = ibo
        self.obj_id = obj_id
        self.aabb = aabb  # (8,3) world-space bounding box corners
        self.has_semantics = has_semantics

    def release(self) -> None:
        for x in (self.vao, self.vbo, self.ibo):
            try:
                x.release()
            except Exception:
                pass


class EGLRasterizer:
    """Rasterizes label channels for a static scene. Reused across scenes within
    a worker process (the EGL context is expensive to create, the geometry is not)."""

    def __init__(self, img_w: int, img_h: int, device_index: int = 0) -> None:
        import moderngl

        self.w = img_w
        self.h = img_h
        try:
            self.ctx = moderngl.create_standalone_context(
                backend="egl", device_index=device_index
            )
        except Exception as e:
            # device_index is unsupported on some glcontext builds; a wrong-GPU
            # context still renders correctly, it just adds cross-GPU traffic
            sp.logger.warning(
                f"EGL device_index={device_index} not honoured ({e}); using default device"
            )
            self.ctx = moderngl.create_standalone_context(backend="egl")

        sp.logger.debug(f"EGL context on {self.ctx.info['GL_RENDERER']}")

        self.prog = self.ctx.program(vertex_shader=_INSTANCE_VS, fragment_shader=_INSTANCE_FS)

        # instance pass: depth-tested, writes each object's id
        self._tex_inst = self.ctx.texture((img_w, img_h), 1, dtype="f4")
        self._depth = self.ctx.depth_texture((img_w, img_h))
        self._fbo_inst = self.ctx.framebuffer(
            color_attachments=[self._tex_inst], depth_attachment=self._depth
        )
        # silhouette pass: no depth test, single object, 1 byte per pixel
        self._tex_sil = self.ctx.texture((img_w, img_h), 1, dtype="f1")
        self._fbo_sil = self.ctx.framebuffer(color_attachments=[self._tex_sil])

        self._entries: list[_Entry] = []

    # ------------------------------------------------------------------ geometry

    def clear_geometry(self) -> None:
        for e in self._entries:
            e.release()
        self._entries = []

    def set_geometry(self, meshes: list[tuple[np.ndarray, np.ndarray, int, bool]]) -> None:
        """upload the scene once. meshes: (verts_world, tris, obj_id, has_semantics)"""
        self.clear_geometry()
        with self.ctx:
            for verts_world, tris, obj_id, has_semantics in meshes:
                if len(tris) == 0 or len(verts_world) == 0:
                    continue
                vbo = self.ctx.buffer(np.ascontiguousarray(verts_world, dtype="f4").tobytes())
                ibo = self.ctx.buffer(np.ascontiguousarray(tris, dtype="i4").tobytes())
                vao = self.ctx.vertex_array(self.prog, [(vbo, "3f", "in_pos")], ibo)
                lo = verts_world.min(axis=0)
                hi = verts_world.max(axis=0)
                aabb = np.array(
                    [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])],
                    dtype=np.float64,
                )
                self._entries.append(_Entry(vao, vbo, ibo, int(obj_id), aabb, bool(has_semantics)))
        sp.logger.debug(f"EGL: uploaded {len(self._entries)} meshes")

    @property
    def n_meshes(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------ matrices

    def _projection(self, K: np.ndarray, near: float = 1e-3, far: float = 1e3) -> np.ndarray:
        """OpenCV intrinsics -> GL clip space, with the y flip (GL's origin is
        bottom-left, the image convention's is top-left)"""
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        P = np.zeros((4, 4), dtype=np.float64)
        P[0, 0] = 2 * fx / self.w
        P[0, 2] = 2 * cx / self.w - 1
        P[1, 1] = -2 * fy / self.h
        P[1, 2] = -(2 * cy / self.h - 1)
        P[2, 2] = (far + near) / (far - near)
        P[2, 3] = -2 * far * near / (far - near)
        P[3, 2] = 1.0
        return P

    @staticmethod
    def _view(cam_location: tuple, cam_rotation) -> np.ndarray:
        """world -> camera. cam_rotation is the camera->world scipy Rotation"""
        Rt = cam_rotation.as_matrix().T
        V = np.eye(4, dtype=np.float64)
        V[:3, :3] = Rt
        V[:3, 3] = -Rt @ np.asarray(cam_location, dtype=np.float64)
        return V

    def _mvp(self, cam_location: tuple, cam_rotation, K: np.ndarray) -> np.ndarray:
        return self._projection(K) @ self._view(cam_location, cam_rotation)

    def _screen_bbox(self, aabb: np.ndarray, mvp: np.ndarray) -> tuple[int, int, int, int] | None:
        """conservative screen-space bbox of an AABB, for a bounded readback.
        Returns None if the box is not safely in front of the camera."""
        h = np.hstack([aabb, np.ones((len(aabb), 1))])
        clip = (mvp @ h.T).T
        w = clip[:, 3]
        if np.any(w <= 1e-9):
            return None  # straddles the camera plane: fall back to a full read
        ndc = clip[:, :3] / w[:, None]
        xs = (ndc[:, 0] * 0.5 + 0.5) * self.w
        ys = (0.5 - ndc[:, 1] * 0.5) * self.h  # ndc y-up -> image y-down
        x0 = int(np.floor(xs.min())) - 1
        x1 = int(np.ceil(xs.max())) + 1
        y0 = int(np.floor(ys.min())) - 1
        y1 = int(np.ceil(ys.max())) + 1
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, self.w - 1), min(y1, self.h - 1)
        if x1 < x0 or y1 < y0:
            return None
        return x0, y0, x1, y1

    # ------------------------------------------------------------------ passes

    def render_instance_mask(self, cam_location: tuple, cam_rotation, K: np.ndarray) -> np.ndarray:
        """depth-tested id map == the 'visib' channel. Every mesh participates as
        an occluder (a mesh with id 0, e.g. the floor plane, occludes but labels
        as background), which is what the EEVEE visib layer did."""
        mvp = self._mvp(cam_location, cam_rotation, K).astype("f4")
        with self.ctx:
            self._fbo_inst.use()
            self.ctx.enable(self.ctx.DEPTH_TEST)
            self.ctx.disable(self.ctx.CULL_FACE)
            self._fbo_inst.clear(0.0, depth=1.0)
            self.prog["MVP"].write(mvp.T.tobytes())
            for e in self._entries:
                self.prog["obj_id"].value = float(e.obj_id)
                e.vao.render()
            self.ctx.finish()
            raw = self._fbo_inst.read(components=1, dtype="f4")
        img = np.frombuffer(raw, dtype="f4").reshape(self.h, self.w).copy()
        img = np.flipud(img)
        return np.rint(img).astype(np.int32)

    def render_silhouettes(
        self, cam_location: tuple, cam_rotation, K: np.ndarray
    ) -> dict[int, tuple[tuple[int, int, int, int], int]]:
        """per-object *unoccluded* silhouette stats for every semantic object:
        {obj_id: (bbox [x1,y1,x2,y2], px_count)}. Each object is drawn alone with
        depth testing off, and only its own screen-space bbox is read back."""
        mvp64 = self._mvp(cam_location, cam_rotation, K)
        mvp = mvp64.astype("f4")
        out: dict[int, tuple[tuple[int, int, int, int], int]] = {}
        with self.ctx:
            self._fbo_sil.use()
            self.ctx.disable(self.ctx.DEPTH_TEST)
            self.ctx.disable(self.ctx.CULL_FACE)
            self.prog["MVP"].write(mvp.T.tobytes())
            for e in self._entries:
                if not e.has_semantics:
                    continue
                self._fbo_sil.clear(0.0)
                self.prog["obj_id"].value = 1.0
                e.vao.render()
                self.ctx.finish()

                roi = self._screen_bbox(e.aabb, mvp64)
                if roi is None:
                    raw = self._fbo_sil.read(components=1, dtype="f1")
                    img = np.flipud(np.frombuffer(raw, dtype=np.uint8).reshape(self.h, self.w))
                    ox, oy = 0, 0
                else:
                    x0, y0, x1, y1 = roi
                    rw, rh = x1 - x0 + 1, y1 - y0 + 1
                    # GL viewport y is measured from the bottom
                    gl_y0 = self.h - (y1 + 1)
                    raw = self._fbo_sil.read(
                        viewport=(x0, gl_y0, rw, rh), components=1, dtype="f1"
                    )
                    img = np.flipud(np.frombuffer(raw, dtype=np.uint8).reshape(rh, rw))
                    ox, oy = x0, y0

                ys, xs = np.nonzero(img)
                if len(ys) == 0:
                    out[e.obj_id] = ((0, 0, 0, 0), 0)
                else:
                    out[e.obj_id] = (
                        (
                            int(xs.min()) + ox,
                            int(ys.min()) + oy,
                            int(xs.max()) + ox,
                            int(ys.max()) + oy,
                        ),
                        int(len(ys)),
                    )
        return out


_rasterizer: EGLRasterizer | None = None


def get_rasterizer(img_w: int, img_h: int) -> EGLRasterizer:
    """process-wide rasterizer; the EGL context is created once and reused across
    scenes. The device comes from SIMPOSE_EGL_DEVICE (set per worker), because
    CUDA_VISIBLE_DEVICES does not constrain EGL device enumeration."""
    global _rasterizer
    if _rasterizer is not None and (_rasterizer.w, _rasterizer.h) == (img_w, img_h):
        return _rasterizer
    if _rasterizer is not None:
        _rasterizer.clear_geometry()
    device_index = int(os.environ.get("SIMPOSE_EGL_DEVICE", "0"))
    _rasterizer = EGLRasterizer(img_w, img_h, device_index=device_index)
    return _rasterizer
