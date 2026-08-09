import contextlib
import yaml
import numpy as np
import time
from pathlib import Path
from PIL import Image
from scipy import ndimage
from scipy.spatial.transform import Rotation as R
from .exr import EXR
import simpose as sp

from typing import TYPE_CHECKING, Callable, ContextManager

if TYPE_CHECKING:
    import bpy


class Scene(sp.observers.Observable):
    def __init__(
        self,
        bl_scene: "bpy.types.Scene|None" = None,
        img_h: int | None = None,
        img_w: int | None = None,
    ) -> None:
        sp.observers.Observable.__init__(self)

        import bpy

        if bl_scene is not None:
            self._bl_scene: bpy.types.Scene = bl_scene
        else:
            assert (
                img_h is not None and img_w is not None
            ), "Must specify resolution when creating a scene"
            self.reset()
            self._initialize(img_h, img_w)
            sp.logger.debug(f"Created scene: {self._bl_scene}")

    def _initialize(self, img_h: int = 480, img_w: int = 640) -> None:
        import bpy, pybullet as p

        self._bl_scene = scene = bpy.data.scenes["Scene"]
        scene["id_counter"] = 0

        # delete cube, light, camera
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)

        # create a lights collection
        scene.collection.children.link(bpy.data.collections.new("Lights"))
        scene.collection.children.link(bpy.data.collections.new("Cameras"))
        scene.collection.children.link(bpy.data.collections.new("Objects"))

        # setup settings
        scene.render.resolution_x = img_w
        scene.render.resolution_y = img_h
        scene.render.resolution_percentage = 100
        scene.render.use_persistent_data = True

        self.output_dir = Path("output")

        self._setup_rendering_device()
        self._setup_compositor()

        # p.connect(p.GUI)  # activate if you want to observe the physics simulation
        p.connect(p.DIRECT)

        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)
        p.setPhysicsEngineParameter(fixedTimeStep=1 / 240.0, numSubSteps=1)

        # disable mouse interaction
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        self.notify(sp.observers.Event.ON_SCENE_CREATED)

    @property
    def resolution(self) -> np.ndarray:
        return np.array((self._bl_scene.render.resolution_x, self._bl_scene.render.resolution_y))

    @property
    def resolution_x(self) -> int:
        return self._bl_scene.render.resolution_x

    @property
    def resolution_y(self) -> int:
        return self._bl_scene.render.resolution_y

    def reset(self) -> None:
        """completely resets the scene"""
        import bpy, pybullet as p

        bpy.ops.wm.read_factory_settings(use_empty=True)

        try:
            p.resetSimulation()
            p.setGravity(0, 0, -9.81)
            p.setRealTimeSimulation(0)
            p.setPhysicsEngineParameter(fixedTimeStep=1 / 240.0, numSubSteps=1)
        except p.error:
            pass

    def step_physics(self, dt: float, apply: bool = True) -> None:
        """steps dt seconds of physics simulation in 1/240s substeps; with apply=False
        the poses are not synced back to Blender (call _apply_simulation once afterwards)"""
        import pybullet as p

        sp.logger.debug(f"Stepping physics for {dt} seconds")
        self.notify(sp.observers.Event.BEFORE_PHYSICS_STEP)

        num_steps = np.floor(240 * dt).astype(int)
        for _ in range(max(1, num_steps)):
            p.stepSimulation()
        # now apply transform to objects
        if apply:
            self._apply_simulation()
        self.notify(sp.observers.Event.AFTER_PHYSICS_STEP)

    def run_simulation(self, with_export: bool = False) -> None:
        import pybullet as p

        p.setRealTimeSimulation(1)
        while True:
            time.sleep(1 / 60.0)
            if not with_export:
                continue
            self._apply_simulation()

    def _apply_simulation(self) -> None:
        """applies pose of physics simulation to blender scene"""
        for obj in self.get_active_objects():
            try:
                obj.apply_pybullet_pose()
            except KeyError:
                pass

    def render(self, gpu_semaphore: ContextManager = contextlib.nullcontext()) -> sp.RenderProduct:
        """renders a single frame and returns a RenderProduct object"""
        index = self._bl_scene.frame_current
        return self.render_batch(np.array([index]), gpu_semaphore)[0]

    def render_batch(
        self,
        indices: np.ndarray,
        gpu_semaphore: ContextManager = contextlib.nullcontext(),
        on_frame_rendered: Callable[[int, sp.RenderProduct], None] | None = None,
    ) -> list[sp.RenderProduct]:
        """Renders frames of a static scene (objects must not move between frames -
        only camera/appearance/light/background may vary).

        Blender renders only the Cycles beauty pass (RGB + depth in one call); both
        label channels come from the EGL rasterizer instead. So the render engine is
        never switched, which lets Cycles' `use_persistent_data` cache stay warm for
        the whole batch, and the label passes run inside the same GPU semaphore hold
        as the Cycles render - one acquire/release per frame, not two.

        Scene geometry is uploaded to the rasterizer once for the whole batch; only
        the view-projection matrix changes per frame.

        on_frame_rendered, if given, is called with (dataset_index, RenderProduct) as
        soon as each frame is ready - e.g. to write it out and tick progress
        immediately, instead of waiting for the whole batch to finish."""
        if gpu_semaphore is None:  # a Writer without a comm has no semaphore
            gpu_semaphore = contextlib.nullcontext()
        camera = self.get_cameras()[0]

        self._compact_object_ids()
        self._set_cycles_render_state()
        rasterizer = self._upload_geometry()

        products = []
        for dataset_index in indices:
            idx = int(dataset_index)
            t0 = time.perf_counter()
            self.frame_set(idx)
            self.notify(sp.observers.Event.BEFORE_RENDER)
            t_randomized = time.perf_counter()

            K = camera.calculate_intrinsics(self.resolution_x, self.resolution_y)
            with gpu_semaphore:
                t_acquired = time.perf_counter()
                self._render_cycles(camera)
                t_cycles = time.perf_counter()
                cam_loc, cam_rot = camera.location, camera.rotation
                mask = rasterizer.render_instance_mask(cam_loc, cam_rot, K)
                sil_stats = rasterizer.render_silhouettes(cam_loc, cam_rot, K)
                t_labels = time.perf_counter()

            self.notify(sp.observers.Event.AFTER_RENDER)
            product = self._build_render_product(idx, camera, mask, sil_stats)
            sp.logger.debug(
                f"frame {idx}: randomize {t_randomized - t0:.2f}s, "
                f"gpu wait {t_acquired - t_randomized:.2f}s, "
                f"cycles {t_cycles - t_acquired:.2f}s, "
                f"labels(egl) {t_labels - t_cycles:.2f}s, "
                f"readback {time.perf_counter() - t_labels:.2f}s"
            )
            products.append(product)
            if on_frame_rendered is not None:
                on_frame_rendered(idx, product)
        return products

    def _compact_object_ids(self) -> None:
        """Renumber the objects' semantic ids to a contiguous 1..N.

        `get_new_object_id` hands out ids from a monotonic per-scene counter, and an
        object that is created and then removed still consumes one. Dropjects
        rejection-samples on object diameter, so a scene of 50 objects can easily end
        up with ids far above 50 - past the 255 the uint8 mask can hold.

        Ids only have to be consistent within a scene, and both the mask and the
        annotations read `pass_index`, so compacting here keeps them in agreement.
        This is only safe because ids no longer name EEVEE view layers."""
        objects = self._bl_scene.collection.children["Objects"].objects
        for new_id, bl_obj in enumerate(objects, start=1):
            bl_obj.pass_index = new_id
        self._bl_scene["id_counter"] = len(objects)

    def _upload_geometry(self) -> "sp.EGLRasterizer":
        """push the current (post-physics) scene geometry to the EGL rasterizer in
        world space. Every visible mesh participates as an occluder - including the
        floor plane, which labels as background (id 0) but still occludes - matching
        what the EEVEE visib layer used to produce."""
        import bpy

        # matrix_world is stale until the depsgraph re-evaluates: setting .location /
        # .rotation_quaternion / .scale does not update it on its own
        bpy.context.view_layer.update()

        meshes = []
        for bl_obj in self._bl_scene.objects:
            if bl_obj.type != "MESH" or bl_obj.hide_render:
                continue
            mesh = bl_obj.data
            mesh.calc_loop_triangles()
            if len(mesh.loop_triangles) == 0:
                continue

            verts = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
            mesh.vertices.foreach_get("co", verts)
            verts = verts.reshape(-1, 3)
            tris = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int32)
            mesh.loop_triangles.foreach_get("vertices", tris)
            tris = tris.reshape(-1, 3)

            mw = np.array(bl_obj.matrix_world, dtype=np.float64)
            verts_world = (mw @ np.hstack([verts, np.ones((len(verts), 1))]).T).T[:, :3]

            meshes.append(
                (verts_world, tris, int(bl_obj.pass_index), bool(bl_obj.get("semantics", False)))
            )

        rasterizer = sp.get_rasterizer(self.resolution_x, self.resolution_y)
        rasterizer.set_geometry(meshes)
        return rasterizer

    def _build_render_product(
        self,
        index: int,
        cam: sp.entities.Camera,
        mask: np.ndarray | None = None,
        sil_stats: dict[int, tuple[tuple[int, int, int, int], int]] | None = None,
    ) -> sp.RenderProduct:
        """assembles the RenderProduct from the Cycles outputs on disk plus the
        label channels produced by the EGL rasterizer (`mask` is the instance id
        map, `sil_stats` the per-object unoccluded bbox/pixel-count)"""
        rgb_path = Path(self.output_dir, "rgb", f"rgb_{index:04}.png")
        rgb_R_path = Path(self.output_dir, "rgb", f"rgb_{index:04}_R.png")
        depth_path = Path(self.output_dir, "depth", f"depth_{index:04}.exr")
        depth_R_path = Path(self.output_dir, "depth", f"depth_{index:04}_R.exr")

        rgb = None if not rgb_path.exists() else self._read_rgb(rgb_path)
        rgb_R = None if not rgb_R_path.exists() else self._read_rgb(rgb_R_path)

        depth = None if not depth_path.exists() else self._read_depth(depth_path)
        depth_R = None if not depth_R_path.exists() else self._read_depth(depth_R_path)

        if sil_stats is None:
            sil_stats = {}

        cam_pos = cam.location
        cam_rot = cam.rotation.as_quat()
        cam_matrix = cam.calculate_intrinsics(self.resolution_x, self.resolution_y)

        obj_annotations: list[sp.ObjectAnnotation] = []
        objs = self.get_active_objects()

        # visib-mask stats (px_count_visib, bbox_visib, px_count_valid) for every
        # object in one vectorized pass instead of one full-frame np.nonzero per
        # object: mask is a per-pixel object-id label image, so a single bincount
        # (counts, and depth-valid counts) plus one find_objects call (bboxes)
        # gives every object's stats at once. Equivalent to the old per-object
        # nonzero/count_nonzero loop, ~15x faster at 50 objects / 2560x1440.
        visib_counts = valid_counts = None
        bbox_slices: list | None = None
        if mask is not None:
            ids = [obj.object_id for obj in objs]
            max_label = int(mask.max())
            if ids:
                max_label = max(max_label, max(ids))
            mask_int = mask.astype(np.int64)
            visib_counts = np.bincount(mask_int.ravel(), minlength=max_label + 1)
            if depth is not None:
                valid_counts = np.bincount(
                    mask_int.ravel(),
                    weights=(depth.ravel() != 0).astype(np.float64),
                    minlength=max_label + 1,
                )
            bbox_slices = ndimage.find_objects(mask_int, max_label=max_label)

        for obj in objs:
            if visib_counts is None:
                px_count_visib = 0
                bbox_visib = (0, 0, 0, 0)
            else:
                oid = obj.object_id
                px_count_visib = int(visib_counts[oid]) if oid < len(visib_counts) else 0
                sl = bbox_slices[oid - 1] if bbox_slices and 0 < oid <= len(bbox_slices) else None
                if sl is None:
                    bbox_visib = (0, 0, 0, 0)
                else:
                    y_sl, x_sl = sl
                    bbox_visib = (x_sl.start, y_sl.start, x_sl.stop - 1, y_sl.stop - 1)
            bbox_obj = (0, 0, 0, 0)
            px_count_all = 0
            px_count_valid = 0
            visib_fract = 0.0

            if obj.has_semantics:
                # unoccluded silhouette, rasterized per object by the EGL pass
                bbox_obj, px_count_all = sil_stats.get(obj.object_id, ((0, 0, 0, 0), 0))
                if valid_counts is not None:
                    oid = obj.object_id
                    px_count_valid = int(valid_counts[oid]) if oid < len(valid_counts) else 0
                if px_count_all != 0:
                    visib_fract = px_count_visib / px_count_all

            annotation = sp.ObjectAnnotation(
                cls=obj.get_class(),
                object_id=obj.object_id,
                position=obj.location,
                quat_xyzw=obj.rotation.as_quat(canonical=False),
                bbox_visib=bbox_visib,
                bbox_obj=bbox_obj,
                px_count_visib=px_count_visib,
                px_count_valid=px_count_valid,
                px_count_all=px_count_all,
                visib_fract=visib_fract,
            )
            obj_annotations.append(annotation)

        stereo_baseline = None if not cam.is_stereo_camera() else cam.baseline

        self.remove_temporary_files(index)  # here?
        return sp.RenderProduct(
            rgb=rgb,
            rgb_R=rgb_R,
            depth=depth,
            depth_R=depth_R,
            mask=mask,
            objs=obj_annotations,
            cam_position=cam_pos,
            cam_quat_xyzw=cam_rot,
            intrinsics=cam_matrix,
            stereo_baseline=stereo_baseline,
        )

    @staticmethod
    def _read_depth(depth_path: Path) -> np.ndarray:
        # blender names the channel of single-channel ("BW") EXRs "V".
        # the EXR is half float; compare/clamp in float32 to avoid the 65504 overflow
        # of half, then hand back float16 -- what the writer stores.
        depth = EXR(depth_path).read("V").astype(np.float32)
        depth[depth > 100.0] = 0.0
        return depth.astype(np.float16)

    def _read_rgb(self, rgb_path: Path) -> np.ndarray:
        """read a composited RGB png, cropped to the render resolution.

        The background Scale node ("Render Size") scales an arbitrary-aspect
        background to *cover* the render, and the composited canvas can end up 1-2 px
        larger than resolution_x/y depending on that particular image's scale factor.
        Observed on a 2560x1440 run: rgb came out (1441, 2560) for one background and
        (1440, 2562) for the next, while depth and mask were always exact -- i.e. RGB
        silently misaligned with the labels *and* with the intrinsics, by a per-frame
        amount. Crop here so every channel and cx/cy agree by construction.
        """
        rgb = np.array(Image.open(str(rgb_path)))
        h, w = self.resolution_y, self.resolution_x
        if rgb.shape[0] < h or rgb.shape[1] < w:
            raise RuntimeError(
                f"{rgb_path.name} is {rgb.shape[1]}x{rgb.shape[0]}, smaller than the "
                f"render resolution {w}x{h}; cannot align it with the labels."
            )
        if rgb.shape[0] != h or rgb.shape[1] != w:
            sp.logger.debug(f"cropping {rgb_path.name} {rgb.shape[1]}x{rgb.shape[0]} -> {w}x{h}")
        return rgb[:h, :w]

    def remove_temporary_files(self, dataset_index: int) -> None:
        output_dir = self.output_dir

        rgb_path = output_dir / "rgb" / f"rgb_{dataset_index:04}.png"
        if rgb_path.exists():
            sp.logger.debug(f"Removing {rgb_path}")
            rgb_path.unlink()

        rgb_R_path = output_dir / "rgb" / f"rgb_{dataset_index:04}_R.png"
        if rgb_R_path.exists():
            sp.logger.debug(f"Removing {rgb_R_path}")
            rgb_R_path.unlink()

        depth_path = output_dir / "depth" / f"depth_{dataset_index:04}.exr"
        if depth_path.exists():
            sp.logger.debug(f"Removing {depth_path}")
            depth_path.unlink()

        depth_R_path = output_dir / "depth" / f"depth_{dataset_index:04}_R.exr"
        if depth_R_path.exists():
            sp.logger.debug(f"Removing {depth_R_path}")
            depth_R_path.unlink()

    def _set_cycles_render_state(self) -> None:
        """Cycles renders RGB + depth. The label channels come from the EGL
        rasterizer, so the engine is set once and never switched again - which is
        what keeps `use_persistent_data` effective across frames."""
        self._bl_scene.render.engine = "CYCLES"
        self.rgb_output.mute = False
        self.depth_output.mute = False

    def _render_cycles(self, camera: sp.entities.Camera) -> None:
        """RGB + depth in one render call (both are outputs of the same pass).
        The caller must hold the GPU semaphore."""
        import bpy

        if camera.is_stereo_camera():
            self._bl_scene.camera = camera.right_camera
            with sp.redirect_stdout():
                sp.logger.debug(f"Rendering right to {self.output_dir}")
                bpy.ops.render.render(write_still=False)
            # rename rendered depth and rgb with suffix _R
            rgb_path = self.output_dir / "rgb" / f"rgb_{self._bl_scene.frame_current:04}.png"
            rgb_path.rename(rgb_path.parent / f"rgb_{self._bl_scene.frame_current:04}_R.png")
            depth_path = (
                self.output_dir / "depth" / f"depth_{self._bl_scene.frame_current:04}.exr"
            )
            depth_path.rename(
                depth_path.parent / f"depth_{self._bl_scene.frame_current:04}_R.exr"
            )

        # for left image and the labels
        self._bl_scene.camera = camera.left_camera
        with sp.redirect_stdout():
            sp.logger.debug(f"Rendering left to {self.output_dir}")
            bpy.ops.render.render(write_still=False)

    def get_new_object_id(self) -> int:
        assert isinstance(self._bl_scene["id_counter"], int)
        self._bl_scene["id_counter"] += 1
        return self._bl_scene["id_counter"]

    def frame_set(self, frame_num: int) -> None:
        self._bl_scene.frame_set(frame_num)  # this sets the suffix for file names

    @property
    def _compositor_tree(self) -> "bpy.types.NodeTree":
        return self._bl_scene.compositing_node_group

    def set_output_path(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.rgb_output.directory = str((output_dir / "rgb").resolve())
        self.depth_output.directory = str((output_dir / "depth").resolve())

    def create_plane(self, size: float = 2, with_physics: bool = True) -> sp.entities.Plane:
        import bpy

        plane = sp.entities.Plane.create(size, with_physics)
        # bpy.data.collections["Objects"].objects.link(plane._bl_object)
        return plane

    def get_cameras(self) -> list[sp.entities.Camera]:
        return [
            sp.entities.Camera(x) for x in self._bl_scene.collection.children["Cameras"].objects
        ]

    def create_camera(self, cam_name: str) -> sp.entities.Camera:
        import bpy

        cam = sp.entities.Camera.create(cam_name, baseline=None)
        bpy.data.collections["Cameras"].objects.link(cam._bl_object)
        return cam

    def create_stereo_camera(self, cam_name: str, baseline: float) -> sp.entities.Camera:
        import bpy

        cam = sp.entities.Camera.create(cam_name, baseline=baseline)
        bpy.data.collections["Cameras"].objects.link(cam._bl_object)
        return cam

    def get_active_objects(self) -> list[sp.entities.Object]:
        objects = [
            sp.entities.Object(x) for x in self._bl_scene.collection.children["Objects"].objects
        ]
        return list([x for x in objects if not x.is_hidden])

    def get_labelled_objects(self) -> list[sp.entities.Object]:
        return list([x for x in self.get_active_objects() if x.has_semantics])

    def create_object(
        self,
        obj_path: Path,
        obj_name: str,
        add_semantics: bool = False,
        mass: float | None = None,
        friction: float = 0.5,
        scale: float = 1.0,
        restitution: float = 0.99,
        hide: bool = False,
    ) -> sp.entities.Object:
        import bpy

        obj_path = obj_path.expanduser()

        if (obj_path.parent / "6IMPOSE_META").exists():
            # overwrite args with meta file
            sp.logger.debug(f"Found META file for {obj_path}")
            meta_path = obj_path.parent / "6IMPOSE_META"
            with open(meta_path, "r") as f:
                meta = yaml.safe_load(f)
            if "mass" in meta:
                mass = float(meta["mass"])
            if "friction" in meta:
                friction = float(meta["friction"])
            if "scale" in meta:
                scale = float(meta["scale"])
            if "name" in meta:
                obj_name = meta["name"]
            if "restitution" in meta:
                restitution = float(meta["restitution"])

        kwargs = dict(
            filepath=obj_path,
            name=obj_name,
            add_semantics=add_semantics,
            mass=mass,
            friction=friction,
            scale=scale,
            restitution=restitution,
        )

        if obj_path.suffix == ".obj":
            obj = sp.entities.Object.from_obj(**kwargs)
        elif obj_path.suffix == ".ply":
            obj = sp.entities.Object.from_ply(**kwargs)
        elif obj_path.suffix == ".gltf":
            obj = sp.entities.Object.from_gltf(**kwargs)
        elif obj_path.suffix == ".fbx":
            obj = sp.entities.Object.from_fbx(**kwargs)
        else:
            raise NotImplementedError(f"Unsupported file format: {obj_path.suffix}")

        new_id = self.get_new_object_id()
        obj.set_semantic_id(new_id)

        if hide:
            obj.hide()

        sp.logger.debug(f"Added {obj.get_name()} to scene.")

        bpy.data.collections["Objects"].objects.link(obj._bl_object)

        self.notify(sp.observers.Event.ON_OBJECT_CREATED)
        return obj

    def create_copy(self, object: sp.entities.Object) -> sp.entities.Object:
        obj = object.copy()
        new_id = self.get_new_object_id()
        obj.set_semantic_id(new_id)

        self.notify(sp.observers.Event.ON_OBJECT_CREATED)
        return obj

    def create_light(
        self, light_name: str, energy: float, type: str = "POINT"
    ) -> sp.entities.Light:
        import bpy

        light = sp.entities.Light.create(light_name, energy, type)
        bpy.data.collections["Lights"].objects.link(light._bl_object)
        return light

    def get_lights(self) -> list[sp.entities.Light]:
        return [sp.entities.Light(x) for x in self._bl_scene.collection.children["Lights"].objects]

    def _setup_rendering_device(self) -> None:
        import bpy

        self._bl_scene.cycles.device = "GPU"
        pref = bpy.context.preferences.addons["cycles"].preferences
        pref.get_devices()  # type: ignore

        for dev in pref.devices:  # type: ignore
            dev.use = False

        device_types = list({x.type for x in pref.devices})  # type: ignore
        priority_list = ["OPTIX", "HIP", "METAL", "ONEAPI", "CUDA"]

        chosen_type = "NONE"

        for type in priority_list:
            if type in device_types:
                chosen_type = type
                break

        sp.logger.info("Rendering device: " + chosen_type)
        # Set GPU rendering mode to detected one
        pref.compute_device_type = chosen_type  # type: ignore

        chosen_type_device = "CPU" if chosen_type == "NONE" else chosen_type
        available_devices = [x for x in pref.devices if x.type == chosen_type_device]  # type: ignore

        selected_devices = [0]  # TODO parametrize this
        for i, dev in enumerate(available_devices):
            if i in selected_devices:
                dev.use = True

        sp.logger.debug(f"Available devices: {available_devices}")

        if chosen_type == "OPTIX":
            self._bl_scene.cycles.denoiser = "OPTIX"
        else:
            self._bl_scene.cycles.denoiser = "OPENIMAGEDENOISE"

    def set_background(self, filepath: Path) -> None:
        import bpy

        # get composition node_tree
        try:
            img = self._bl_scene["background_img"]
            bpy.data.images.remove(img)
        except KeyError:
            pass
        # if self.current_bg_img is not None:
        #    bpy.data.images.remove(self.current_bg_img)
        new_img = bpy.data.images.load(str(filepath.resolve()))

        tree = self._compositor_tree
        bg_image_node: bpy.types.CompositorNodeImage = tree.nodes["background_node"]  # type: ignore
        bg_image_node.image = new_img
        self._bl_scene["background_img"] = new_img
        # scale_to_fit = np.max(self.resolution / np.array(self.current_bg_img.size))
        sp.logger.debug(f"Set background to {filepath}")

    def export_blend(self, filepath: Path = Path("scene.blend")) -> None:
        import bpy

        # import simpose.register_addon

        # this does not work atm
        # register_script = simpose.register_addon.__file__
        # bpy.ops.script.python_file_run(filepath=str(register_script))

        with sp.redirect_stdout():
            bpy.ops.wm.save_as_mainfile(filepath=str(filepath.resolve()))
        sp.logger.debug(f"Export scene to {filepath.resolve()}")

    def export_meshes(self, output_dir: Path) -> None:
        """export meshes as ply files in 'meshes' folder"""
        objs = {obj.get_class(): obj for obj in self.get_labelled_objects()}
        for obj in objs.values():
            obj.export_as_ply(output_dir)

    def _setup_compositor(self) -> None:
        import bpy

        self._bl_scene.render.film_transparent = True
        self._bl_scene.render.use_simplify = True

        # no tone mapping (Filmic/AgX): composited real backgrounds must round-trip
        # and output must not depend on the installed bpy version's default transform
        self._bl_scene.view_settings.view_transform = "Standard"

        self._bl_scene.view_layers["ViewLayer"].use_pass_z = True
        self._bl_scene.view_layers["ViewLayer"].use_pass_combined = True
        self._bl_scene.render.engine = "CYCLES"
        self._bl_scene.cycles.use_denoising = True
        self._bl_scene.cycles.use_preview_denoising = True
        self._bl_scene.cycles.samples = 64
        self._bl_scene.cycles.preview_samples = 64
        self._bl_scene.cycles.use_auto_tile = False
        # self._bl_scene.cycles.tile_size = 256
        self._bl_scene.cycles.caustics_reflective = True
        self._bl_scene.cycles.caustics_refractive = True
        self._bl_scene.cycles.use_camera_cull = True
        self._bl_scene.cycles.use_distance_cull = True

        self._bl_scene.eevee.taa_render_samples = 1
        self._bl_scene.eevee.taa_samples = 1

        # set number of bounces
        self._bl_scene.cycles.max_bounces = 4
        self._bl_scene.cycles.diffuse_bounces = 3
        self._bl_scene.cycles.glossy_bounces = 3
        self._bl_scene.cycles.transparent_max_bounces = 4

        # since blender 5.0, scene compositing lives in a node group datablock
        tree = bpy.data.node_groups.new("sp_compositor", "CompositorNodeTree")
        self._bl_scene.compositing_node_group = tree

        # clear node tree
        for node in tree.nodes:
            tree.nodes.remove(node)

        # add render layers node
        self.render_layers = render_layers = tree.nodes.new("CompositorNodeRLayers")

        # create a alpha node to overlay the rendered image over the background image
        self.alpha_over: bpy.types.CompositorNodeAlphaOver = tree.nodes.new(
            "CompositorNodeAlphaOver"
        )  # type: ignore
        self.alpha_over.location = (600, 300)

        bg_image_node: bpy.types.CompositorNodeImage = tree.nodes.new("CompositorNodeImage")  # type: ignore
        bg_image_node.location = (0, 300)
        bg_image_node.name = "background_node"

        # add scale node
        scale_node: bpy.types.CompositorNodeScale = tree.nodes.new("CompositorNodeScale")  # type: ignore
        scale_node.inputs["Type"].default_value = "Render Size"
        scale_node.location = (300, 300)

        # bg node -> scale
        tree.links.new(bg_image_node.outputs[0], scale_node.inputs["Image"])
        # scale -> alpha over background
        tree.links.new(scale_node.outputs[0], self.alpha_over.inputs["Background"])
        # rendered rgb (premultiplied alpha via film_transparent) -> foreground;
        # the foreground's own alpha channel drives the compositing
        tree.links.new(render_layers.outputs["Image"], self.alpha_over.inputs["Foreground"])

        # RGB image output -> rgb/rgb_{frame:04}.png
        self.rgb_output = self._create_image_output(tree, "rgb", "RGBA")
        self.rgb_output.location = (900, 0)
        self.rgb_output.format.file_format = "PNG"
        self.rgb_output.format.color_mode = "RGB"
        tree.links.new(self.alpha_over.outputs[0], self.rgb_output.inputs[0])

        # Depth output -> depth/depth_{frame:04}.exr
        # single channel, full float32: half floats quantize depth GT to mm-level steps
        self.depth_output = self._create_image_output(tree, "depth", "FLOAT")
        self.depth_output.location = (900, -250)
        self.depth_output.format.file_format = "OPEN_EXR"
        self.depth_output.format.color_mode = "BW"
        self.depth_output.format.exr_codec = "ZIP"
        # half float: ~1 mm quantization over this pipeline's 0.5-2.5 m working range,
        # comparable to a real depth sensor, at half the bytes of "32". float32 depth
        # made a datapoint 28.6 MB vs 11.6 MB for the 16-bit omni run, because float32
        # depth also compresses much worse.
        self.depth_output.format.color_depth = "16"
        tree.links.new(render_layers.outputs["Depth"], self.depth_output.inputs[0])

        # no mask output: the instance mask and the per-object unoccluded
        # silhouettes are rasterized by the EGL rasterizer instead of being
        # rendered by EEVEE into a multilayer EXR

        self.set_output_path(self.output_dir)

    def _create_image_output(
        self, tree: "bpy.types.NodeTree", name: str, socket_type: str
    ) -> "bpy.types.CompositorNodeOutputFile":
        """File Output node writing single images to <dir>/{name}_{frame:04}.<ext>;
        the directory is set via set_output_path, the format by the caller"""
        import bpy

        node: bpy.types.CompositorNodeOutputFile = tree.nodes.new(
            "CompositorNodeOutputFile"
        )  # type: ignore
        node.format.media_type = "IMAGE"  # unlocks non-multilayer formats
        node.file_name = f"{name}_####"  # #### = current frame
        node.file_output_items.new(socket_type, "")
        return node
