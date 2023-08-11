from typing import Dict
from .redirect_stdout import redirect_stdout

with redirect_stdout():
    import bpy
    import pybullet as p
    import pybullet_data

from simpose.camera import Camera
from simpose.light import Light
from simpose.object import Object
import numpy as np
import logging
from typing import List, Tuple
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from simpose.callback import Callback, Callbacks, CallbackType


class Scene(Callbacks):
    def __init__(self, img_h: int = 480, img_w: int = 640, use_stereo: bool = False) -> None:
        Callbacks.__init__(self)
        # self._bl_scene = bpy.data.scenes.new("6impose Scene")
        self._bl_scene: bpy.types.Scene = bpy.context.window.scene

        # delete old cup, light, camera and scene
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)

        # bpy.context.window.scene= self._bl_scene
        self.__id_counter = 0  # never directly access

        self._randomizers: List[Callback] = []

        # add new custom integer property to view layer, called 'index'
        self._bl_scene.view_layers["ViewLayer"]["object_index"] = 0

        # create a lights collection
        self._bl_scene.collection.children.link(bpy.data.collections.new("Lights"))
        self._bl_scene.collection.children.link(bpy.data.collections.new("Cameras"))
        self._bl_scene.collection.children.link(bpy.data.collections.new("Objects"))

        bpy.context.scene.render.use_multiview = use_stereo

        # setup settings
        bpy.context.scene.render.resolution_x = img_w
        bpy.context.scene.render.resolution_y = img_h
        bpy.context.scene.render.resolution_percentage = 100
        bpy.context.scene.render.use_persistent_data = True
        self.resolution = np.array([img_w, img_h])
        bpy.context.scene.view_layers[0].cycles.use_denoising = True

        self.current_bg_img: bpy.types.Image | None = None
        self.output_dir = Path("output")
        self._setup_compositor()
        self._setup_rendering_device()

        p.connect(p.DIRECT)
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)
        # set timestep to 1/24. with 10substeps
        p.setPhysicsEngineParameter(fixedTimeStep=1 / 24.0, numSubSteps=10)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")  # XY ground plane

        self.callback(CallbackType.ON_SCENE_CREATED)

    def step_physics(self, dt):
        """steps 1/240sec of physics simulation"""
        num_steps = np.floor(24 * dt).astype(int)
        for _ in range(max(1, num_steps)):
            p.stepSimulation()
        # now apply transform to objects
        for obj in self.get_active_objects():
            try:
                pb_id = obj._bl_object["pb_id"]
                pos, orn = p.getBasePositionAndOrientation(pb_id)
                obj.set_location(pos)
                obj.set_rotation(R.from_quat(orn))
            except KeyError:
                pass

        self.callback(CallbackType.ON_PHYSICS_STEP)

    def render(self, render_object_masks: bool):
        logging.getLogger().debug("Rendering")
        self.callback(CallbackType.BEFORE_RENDER)
        # render RGB and DEPTH
        bpy.context.scene.render.engine = "CYCLES"
        bpy.context.scene.cycles.use_denoising = True
        bpy.context.scene.cycles.samples = 64
        bpy.context.scene.cycles.caustics_reflective = False
        bpy.context.scene.cycles.caustics_refractive = False
        bpy.context.scene.cycles.use_auto_tile = False
        # set number of bounces
        bpy.context.scene.cycles.max_bounces = 4
        bpy.context.scene.cycles.min_bounces = 0
        bpy.context.scene.cycles.diffuse_bounces = 3
        bpy.context.scene.cycles.glossy_bounces = 3
        bpy.context.scene.cycles.transparent_max_bounces = 4

        bpy.context.view_layer.use_pass_z = True

        tree = bpy.context.scene.node_tree
        output = self.output_node
        output.file_slots[0].path = "rgb/rgb_"
        output.file_slots[0].use_node_format = False
        output.file_slots[0].format.color_mode = "RGB"
        output.file_slots[0].format.file_format = "PNG"
        tree.links.new(self.alpha_over.outputs[0], output.inputs["rgb"])

        with redirect_stdout():
            bpy.ops.render.render(write_still=False)

        # render individual MASKS
        bpy.context.scene.render.engine = "BLENDER_EEVEE"
        self._bl_scene.eevee.taa_render_samples = 1
        self._bl_scene.eevee.taa_samples = 1

        bpy.context.view_layer.use_pass_z = False

        # connect renderlayer directly to output node
        tree = bpy.context.scene.node_tree
        tree.links.new(self.render_layers.outputs[0], self.output_node.inputs["rgb"])

        self._bl_scene.view_layers["ViewLayer"]["object_index"] = 0  # all visible masks
        output = self.output_node
        output.file_slots[0].path = "mask/mask_"
        output.file_slots[0].use_node_format = False
        output.file_slots[0].format.color_mode = "RGB"
        output.file_slots[0].format.file_format = "OPEN_EXR"
        output.file_slots[0].format.exr_codec = "ZIPS"  # lossless
        output.file_slots[0].format.color_depth = "16"
        with redirect_stdout():
            bpy.ops.render.render(write_still=False)

        if render_object_masks:
            for obj in self.get_labelled_objects():
                self._bl_scene.view_layers["ViewLayer"]["object_index"] = obj.object_id
                output.file_slots[0].path = f"mask/mask_{obj.object_id:04d}_"
                with redirect_stdout():
                    bpy.ops.render.render(write_still=False)

        self.callback(CallbackType.AFTER_RENDER)

    def get_new_object_id(self) -> int:
        self.__id_counter += 1
        return self.__id_counter

    def frame_set(self, frame_num: int):
        self._bl_scene.frame_set(frame_num)  # this sets the suffix for file names

    def set_output_path(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_node.base_path = str((self.output_dir).resolve())

    def get_cameras(self) -> List[Camera]:
        return [Camera(x) for x in self._bl_scene.collection.children["Cameras"].objects]

    def create_camera(self, cam_name: str) -> Camera:
        cam = Camera.create(cam_name)
        # add camera to "Cameras" collection
        bpy.data.collections["Cameras"].objects.link(cam._bl_object)
        # remove from default collection
        # bpy.context.scene.collection.objects.unlink(cam._bl_object)
        return cam

    def get_active_objects(self) -> List[Object]:
        objects = [Object(x) for x in self._bl_scene.collection.children["Objects"].objects]
        return list([x for x in objects if not x.is_hidden])

    def get_labelled_objects(self) -> List[Object]:
        return list([x for x in self.get_active_objects() if x.has_semantics])

    def create_object(
        self,
        obj_path: Path,
        add_semantics: bool = False,
        mass: float | None = None,
        friction: float = 0.5,
        scale: float = 1.0,
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
        else:
            raise NotImplementedError("Only .obj and .ply files are supported")

        if add_semantics:
            obj._bl_object.pass_index = self.get_new_object_id()
        bpy.data.collections["Objects"].objects.link(obj._bl_object)

        self.callback(CallbackType.ON_OBJECT_CREATED)
        return obj

    def create_copy(self, object: Object, linked: bool = False) -> Object:
        obj = object.copy(linked=linked)
        if obj.has_semantics:
            # assign new instance id to copy
            obj._bl_object.pass_index = self.get_new_object_id()
        self.callback(CallbackType.ON_OBJECT_CREATED)
        return obj

    def create_light(self, light_name: str, energy: float, type="POINT") -> Light:
        light = Light.create(light_name, energy, type)
        bpy.data.collections["Lights"].objects.link(light._bl_object)
        return light

    def set_gravity(self, gravity: Tuple[float, float, float]):
        bpy.context.scene.gravity = gravity

    def _setup_rendering_device(self):
        bpy.context.scene.cycles.device = "GPU"
        pref = bpy.context.preferences.addons["cycles"].preferences
        pref.get_devices()

        for dev in pref.devices:
            dev.use = False

        device_types = list({x.type for x in pref.devices})
        priority_list = ["OPTIX", "HIP", "METAL", "ONEAPI", "CUDA"]

        chosen_type = "NONE"

        for type in priority_list:
            if type in device_types:
                chosen_type = type
                break

        logging.info("Rendering device: " + chosen_type)
        # Set GPU rendering mode to detected one
        pref.compute_device_type = chosen_type

        chosen_type_device = "CPU" if chosen_type == "NONE" else chosen_type
        available_devices = [x for x in pref.devices if x.type == chosen_type_device]

        selected_devices = [0]  # TODO parametrize this
        for i, dev in enumerate(available_devices):
            if i in selected_devices:
                dev.use = True

        logging.debug(f"Available devices: {available_devices}")

        if chosen_type == "OPTIX":
            bpy.context.scene.cycles.denoiser = "OPTIX"
        else:
            bpy.context.scene.cycles.denoiser = "OPENIMAGEDENOISE"

    def set_background(self, filepath):
        # get composition node_tree
        if self.current_bg_img is not None:
            bpy.data.images.remove(self.current_bg_img)
        self.current_bg_img = bpy.data.images.load(filepath)
        self.bg_image_node.image = self.current_bg_img
        scale_to_fit = np.max(self.resolution / np.array(self.current_bg_img.size))
        self.bg_transform.inputs[4].default_value = scale_to_fit
        logging.debug(f"Set background to {filepath}")

    def export_blend(self, filepath=str(Path("scene.blend").resolve())):
        with redirect_stdout():
            bpy.ops.wm.save_as_mainfile(filepath=filepath)

    def _setup_compositor(self):
        bpy.context.scene.use_nodes = True
        bpy.context.scene.render.film_transparent = True
        bpy.context.view_layer.use_pass_z = True
        bpy.context.view_layer.use_pass_combined = True
        tree = bpy.context.scene.node_tree

        # clear node tree
        for node in tree.nodes:
            tree.nodes.remove(node)

        # add render layers node
        self.render_layers = render_layers = tree.nodes.new("CompositorNodeRLayers")

        # create a alpha node to overlay the rendered image over the background image
        self.alpha_over: bpy.types.CompositorNodeAlphaOver = tree.nodes.new(
            "CompositorNodeAlphaOver"
        )
        self.bg_image_node: bpy.types.CompositorNodeImage = tree.nodes.new("CompositorNodeImage")

        # create a transform node to scale the background image to fit the render resolution
        self.bg_transform: bpy.types.CompositorNodeTransform = tree.nodes.new(
            "CompositorNodeTransform"
        )
        self.bg_transform.filter_type = "BILINEAR"

        # invert alpha
        invert_alpha: bpy.types.CompositorNodeMath = tree.nodes.new("CompositorNodeMath")
        invert_alpha.operation = "SUBTRACT"
        invert_alpha.inputs[0].default_value = 1.0

        # bg_node -> transform
        tree.links.new(self.bg_image_node.outputs[0], self.bg_transform.inputs[0])
        # transform -> alpha over
        tree.links.new(self.bg_transform.outputs[0], self.alpha_over.inputs[2])
        # rendered_rgb -> alpha over
        tree.links.new(render_layers.outputs[0], self.alpha_over.inputs[1])
        # rendered_alpha -> invert alpha
        tree.links.new(render_layers.outputs[1], invert_alpha.inputs[1])
        # invert alpha -> alpha_over
        tree.links.new(invert_alpha.outputs[0], self.alpha_over.inputs[0])

        # RGB image output
        self.output_node: bpy.types.CompositorNodeOutputFile = tree.nodes.new(
            "CompositorNodeOutputFile"
        )
        output: bpy.types.CompositorNodeOutputFile = self.output_node
        output.base_path = str((self.output_dir).resolve())
        output.inputs.remove(output.inputs[0])

        # RGB output
        output.file_slots.new("rgb")
        output.file_slots[0].path = "rgb/rgb_"
        output.file_slots[0].use_node_format = False
        output.file_slots[0].format.color_mode = "RGB"
        output.file_slots[0].format.file_format = "PNG"
        tree.links.new(self.alpha_over.outputs[0], output.inputs["rgb"])

        # Depth output
        output.file_slots.new("depth")
        output.file_slots[1].path = "depth/depth_"
        output.file_slots[1].use_node_format = False
        output.file_slots[1].format.color_mode = "RGB"
        output.file_slots[1].format.file_format = "OPEN_EXR"
        output.file_slots[1].format.exr_codec = "ZIP"
        output.file_slots[1].format.color_depth = "16"
        tree.links.new(render_layers.outputs["Depth"], output.inputs["depth"])
