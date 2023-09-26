from .redirect_stdout import redirect_stdout

import bpy

with redirect_stdout():
    import pybullet as p
    import pybullet_data

from simpose.camera import Camera
from simpose.light import Light
from simpose.object import Object
import numpy as np
import logging
import time
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from simpose.callback import Callback, Callbacks, CallbackType

logger = logging.getLogger("simpose")


class Scene(Callbacks):
    def __init__(self, img_h: int = 480, img_w: int = 640) -> None:
        Callbacks.__init__(self)
        # self._bl_scene = bpy.data.scenes.new("6impose Scene")
        self._bl_scene: bpy.types.Scene = bpy.data.scenes["Scene"]

        scene = self._bl_scene
        self.__id_counter = 0

        # delete old cup, light, camera and scene
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)

        # bpy.context.window.scene= self._bl_scene
        self._randomize: list[Callback] = []

        # create a lights collection
        scene.collection.children.link(bpy.data.collections.new("Lights"))
        scene.collection.children.link(bpy.data.collections.new("Cameras"))
        scene.collection.children.link(bpy.data.collections.new("Objects"))

        # setup settings
        scene.render.resolution_x = img_w
        scene.render.resolution_y = img_h
        scene.render.resolution_percentage = 100
        scene.render.use_persistent_data = False  # HACK
        scene.view_layers[0].cycles.use_denoising = True

        self.current_bg_img: bpy.types.Image | None = None
        self.output_dir = Path("output")
        self._setup_rendering_device()
        self._setup_compositor()
        self._register_new_id("visib")

        if logger.level >= logging.DEBUG:
            p.connect(p.DIRECT)
        else:
            p.connect(p.GUI)

        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)
        # set timestep to 1/24. with 10substeps
        p.setPhysicsEngineParameter(fixedTimeStep=1 / 240.0, numSubSteps=1)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        plane = p.loadURDF("plane.urdf")  # XY ground plane
        p.changeDynamics(plane, -1, lateralFriction=0.5, restitution=0.8)

        self.call_callback(CallbackType.ON_SCENE_CREATED)

    @property
    def resolution(self):
        return np.array((self._bl_scene.render.resolution_x, self._bl_scene.render.resolution_y))

    @property
    def resolution_x(self):
        return self._bl_scene.render.resolution_x

    @property
    def resolution_y(self):
        return self._bl_scene.render.resolution_y

    def step_physics(self, dt):
        """steps 1/240sec of physics simulation"""
        logger.debug(f"Stepping pyhysics for {dt} seconds")
        self.call_callback(CallbackType.BEFORE_PHYSICS_STEP)

        num_steps = np.floor(240 * dt).astype(int)
        for _ in range(max(1, num_steps)):
            p.stepSimulation()
        # now apply transform to objects
        self._apply_simulation()
        self.call_callback(CallbackType.AFTER_PHYSICS_STEP)

    def run_simulation(self, with_export=False):
        p.setRealTimeSimulation(1)
        while True:
            time.sleep(1 / 60.0)
            if not with_export:
                continue
            self._apply_simulation()
            self.export_blend()

    def _apply_simulation(self):
        for obj in self.get_active_objects():
            try:
                obj.apply_pybullet_pose()
            except KeyError:
                pass

    def render(self):
        logger.debug("Rendering")
        self.call_callback(CallbackType.BEFORE_RENDER)

        # render RGB, depth using cycles
        # disable all view layers except 'ViewLayer'
        for layer in self._bl_scene.view_layers:
            layer.use = False
        self._bl_scene.view_layers["ViewLayer"].use = True
        self._bl_scene.render.engine = "CYCLES"
        self.output_node.mute = False
        self.mask_output.mute = True

        camera = self.get_cameras()[0]

        if camera.is_stereo_camera():
            self._bl_scene.camera = camera.right_camera
            with redirect_stdout():
                bpy.ops.render.render(write_still=False)
            # rename rendered depth and rgb with suffix _R
            rgb_path = self.output_dir / "rgb" / f"rgb_{self._bl_scene.frame_current:04}.png"
            rgb_path.rename(rgb_path.parent / f"rgb_{self._bl_scene.frame_current:04}_R.png")
            depth_path = self.output_dir / "depth" / f"depth_{self._bl_scene.frame_current:04}.exr"
            depth_path.rename(depth_path.parent / f"depth_{self._bl_scene.frame_current:04}_R.exr")

        # for left image and the labels
        self._bl_scene.camera = camera.left_camera
        with redirect_stdout():
            logger.debug(f"Rendering left to {self.output_dir}")
            bpy.ops.render.render(write_still=False)

        # render mask into a single EXR using eevee
        # enable all view layers except ViewLayer
        for layer in self._bl_scene.view_layers:
            layer.use = True
        self._bl_scene.view_layers["ViewLayer"].use = False
        self._bl_scene.render.engine = "BLENDER_EEVEE"
        self.output_node.mute = True
        self.mask_output.mute = False
        with redirect_stdout():
            bpy.ops.render.render(write_still=False)

        self.call_callback(CallbackType.AFTER_RENDER)

    def get_new_object_id(self) -> int:
        self.__id_counter += 1
        return self.__id_counter

    def frame_set(self, frame_num: int):
        self._bl_scene.frame_set(frame_num)  # this sets the suffix for file names

    def set_output_path(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_node.base_path = str((self.output_dir).resolve())
        self.mask_output.base_path = str((self.output_dir / "mask/mask_").resolve())

    def get_cameras(self) -> list[Camera]:
        return [Camera(x) for x in self._bl_scene.collection.children["Cameras"].objects]

    def create_camera(self, cam_name: str) -> Camera:
        cam = Camera.create(cam_name, baseline=None)
        bpy.data.collections["Cameras"].objects.link(cam._bl_object)
        return cam

    def create_stereo_camera(self, cam_name: str, baseline: float) -> Camera:
        cam = Camera.create(cam_name, baseline=baseline)
        bpy.data.collections["Cameras"].objects.link(cam._bl_object)
        return cam

    def get_active_objects(self) -> list[Object]:
        objects = [Object(x) for x in self._bl_scene.collection.children["Objects"].objects]
        return list([x for x in objects if not x.is_hidden])

    def get_labelled_objects(self) -> list[Object]:
        return list([x for x in self.get_active_objects() if x.has_semantics])

    def create_object(
        self,
        obj_path: Path,
        add_semantics: bool = False,
        mass: float | None = None,
        friction: float = 0.5,
        scale: float = 1.0,
        hide: bool = False,
    ) -> Object:
        if obj_path.suffix == ".obj":
            obj = Object.from_obj(
                filepath=obj_path,
                add_semantics=add_semantics,
                mass=mass,
                friction=friction,
                scale=scale,
            )
        elif obj_path.suffix == ".ply":
            obj = Object.from_ply(
                filepath=obj_path,
                add_semantics=add_semantics,
                mass=mass,
                friction=friction,
                scale=scale,
            )
        elif obj_path.suffix == ".gltf":
            obj = Object.from_gltf(
                filepath=obj_path,
                add_semantics=add_semantics,
                mass=mass,
                friction=friction,
                scale=scale,
            )
        elif obj_path.suffix == ".fbx":
            obj = Object.from_fbx(
                filepath=obj_path,
                add_semantics=add_semantics,
                mass=mass,
                friction=friction,
                scale=scale,
            )
        else:
            raise NotImplementedError(f"Unsupported file format: {obj_path.suffix}")

        if add_semantics:
            new_id = self.get_new_object_id()
            obj.set_semantic_id(new_id)
            self._register_new_id(new_id)

        if hide:
            obj.hide()

        bpy.data.collections["Objects"].objects.link(obj._bl_object)

        self.call_callback(CallbackType.ON_OBJECT_CREATED)
        return obj

    def create_copy(self, object: Object) -> Object:
        obj = object.copy()
        if obj.has_semantics:
            new_id = self.get_new_object_id()
            obj.set_semantic_id(new_id)
            self._register_new_id(new_id)

        self.call_callback(CallbackType.ON_OBJECT_CREATED)
        return obj

    def _register_new_id(self, new_id: str | int):
        # create a new view layer
        if isinstance(new_id, int):
            id = new_id
            layer_name = f"{new_id:04}"
        else:
            id = 0
            layer_name = new_id

        # new layer with id_material override and object index
        self._bl_scene.view_layers.new(name=layer_name)
        view_layer = self._bl_scene.view_layers[layer_name]
        view_layer["object_index"] = id

        # add file output to compositor
        tree = self._bl_scene.node_tree
        layer_node: bpy.types.CompositorNodeRLayers = tree.nodes.new("CompositorNodeRLayers")  # type: ignore
        layer_node.layer = layer_name
        layer_node.location = (0, -500 - id * 100)

        self.mask_output.file_slots.new(layer_name)
        tree.links.new(layer_node.outputs["Image"], self.mask_output.inputs[layer_name])

    def create_light(self, light_name: str, energy: float, type="POINT") -> Light:
        light = Light.create(light_name, energy, type)
        bpy.data.collections["Lights"].objects.link(light._bl_object)
        return light

    def _setup_rendering_device(self):
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

        logger.info("Rendering device: " + chosen_type)
        # Set GPU rendering mode to detected one
        pref.compute_device_type = chosen_type  # type: ignore

        chosen_type_device = "CPU" if chosen_type == "NONE" else chosen_type
        available_devices = [x for x in pref.devices if x.type == chosen_type_device]  # type: ignore

        selected_devices = [0]  # TODO parametrize this
        for i, dev in enumerate(available_devices):
            if i in selected_devices:
                dev.use = True

        logger.debug(f"Available devices: {available_devices}")

        if chosen_type == "OPTIX":
            self._bl_scene.cycles.denoiser = "OPTIX"
        else:
            self._bl_scene.cycles.denoiser = "OPENIMAGEDENOISE"

    def set_background(self, filepath: Path):
        # get composition node_tree
        if self.current_bg_img is not None:
            bpy.data.images.remove(self.current_bg_img)
        self.current_bg_img = bpy.data.images.load(str(filepath.resolve()))
        self.bg_image_node.image = self.current_bg_img
        # scale_to_fit = np.max(self.resolution / np.array(self.current_bg_img.size))
        logger.debug(f"Set background to {filepath}")

    def export_blend(self, filepath: Path = Path("scene.blend")):
        self._bl_scene.render.engine = "CYCLES"
        with redirect_stdout():
            bpy.ops.wm.save_as_mainfile(filepath=str(filepath.resolve()))
        logger.debug(f"Export scene to {filepath.resolve()}")

    def export_meshes(self, output_dir: Path):
        """export meshes as ply files in 'meshes' folder"""
        for obj in self.get_labelled_objects():
            obj.export_as_ply(output_dir)

    def _setup_compositor(self):
        self._bl_scene.use_nodes = True
        self._bl_scene.render.film_transparent = True
        self._bl_scene.render.use_simplify = True

        self._bl_scene.view_layers["ViewLayer"].use_pass_z = True
        self._bl_scene.view_layers["ViewLayer"].use_pass_combined = True
        self._bl_scene.render.engine = "CYCLES"
        self._bl_scene.cycles.use_denoising = True
        self._bl_scene.cycles.samples = 64
        self._bl_scene.cycles.caustics_reflective = False
        self._bl_scene.cycles.caustics_refractive = False
        self._bl_scene.cycles.use_auto_tile = True
        self._bl_scene.cycles.tile_size = 256
        self._bl_scene.cycles.caustics_reflective = True
        self._bl_scene.cycles.caustics_refractive = True
        self._bl_scene.cycles.use_camera_cull = True
        self._bl_scene.cycles.use_distance_cull = True

        self._bl_scene.eevee.taa_render_samples = 1
        self._bl_scene.eevee.taa_samples = 1

        # set number of bounces
        self._bl_scene.cycles.max_bounces = 4
        self._bl_scene.cycles.min_bounces = 0
        self._bl_scene.cycles.diffuse_bounces = 3
        self._bl_scene.cycles.glossy_bounces = 3
        self._bl_scene.cycles.transparent_max_bounces = 4

        tree = self._bl_scene.node_tree

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

        self.bg_image_node: bpy.types.CompositorNodeImage = tree.nodes.new("CompositorNodeImage")  # type: ignore
        self.bg_image_node.location = (0, 300)

        # add scale node
        scale_node: bpy.types.CompositorNodeScale = tree.nodes.new("CompositorNodeScale")  # type: ignore
        scale_node.space = "RENDER_SIZE"
        scale_node.location = (300, 300)

        # bg node -> scale
        tree.links.new(self.bg_image_node.outputs[0], scale_node.inputs[0])
        # scale -> alpha over [1]
        tree.links.new(scale_node.outputs[0], self.alpha_over.inputs[1])
        # rendered_rgb -> alpha over [2]
        tree.links.new(render_layers.outputs[0], self.alpha_over.inputs[2])
        # rendered_alpha -> alpha over [0]
        tree.links.new(render_layers.outputs[1], self.alpha_over.inputs[0])

        # RGB image output
        self.output_node: bpy.types.CompositorNodeOutputFile = tree.nodes.new(
            "CompositorNodeOutputFile"
        )  # type: ignore
        output: bpy.types.CompositorNodeOutputFile = self.output_node  # type: ignore
        output.base_path = str((self.output_dir).resolve())
        output.inputs.remove(output.inputs[0])
        output.location = (900, 0)

        # RGB output
        output.file_slots.new("rgb")
        output.file_slots[0].path = "rgb/rgb_"
        output.file_slots[0].use_node_format = False
        output.file_slots[0].format.color_mode = "RGB"
        output.file_slots[0].format.file_format = "PNG"
        tree.links.new(self.alpha_over.outputs[0], output.inputs["rgb"])

        # Depth output
        output.file_slots.new("depth")
        t: bpy.types.NodeOutputFileSlotFile = output.file_slots[1]
        output.file_slots[1].path = "depth/depth_"
        output.file_slots[1].use_node_format = False
        output.file_slots[1].format.color_mode = "RGB"
        output.file_slots[1].format.file_format = "OPEN_EXR"
        output.file_slots[1].format.exr_codec = "ZIP"
        output.file_slots[1].format.color_depth = "16"
        tree.links.new(render_layers.outputs["Depth"], output.inputs["depth"])

        # mask output
        self.mask_output: bpy.types.CompositorNodeOutputFile = tree.nodes.new(
            "CompositorNodeOutputFile"
        )  # type: ignore
        mask_output: bpy.types.CompositorNodeOutputFile = self.mask_output  # type: ignore
        mask_output.location = (400, -500)
        mask_output.base_path = str((self.output_dir / "mask/mask_").resolve())
        mask_output.inputs.remove(mask_output.inputs[0])
        mask_output.format.file_format = "OPEN_EXR_MULTILAYER"
        mask_output.format.color_depth = "16"
        mask_output.format.exr_codec = "ZIP"
