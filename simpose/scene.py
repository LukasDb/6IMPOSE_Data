from typing import Dict
from simpose.camera import Camera
from simpose.light import Light
from simpose.object import Object
import bpy
import numpy as np
import logging
from .redirect_stdout import redirect_stdout
from typing import List, Tuple
from pathlib import Path


class Scene:
    def __init__(self, img_h: int = 480, img_w: int = 640) -> None:
        self._bl_scene = bpy.data.scenes.new("6impose Scene")
        bpy.context.window.scene = self._bl_scene
        self.__id_counter = 0  # never access

        # create a lights collection
        self._bl_scene.collection.children.link(bpy.data.collections.new("Lights"))
        self._bl_scene.collection.children.link(bpy.data.collections.new("Cameras"))
        self._bl_scene.collection.children.link(bpy.data.collections.new("Objects"))

        # setup settings
        bpy.context.scene.render.engine = "CYCLES"
        bpy.context.scene.cycles.use_denoising = True
        bpy.context.scene.cycles.samples = 64
        bpy.context.scene.cycles.caustics_reflective = False
        bpy.context.scene.cycles.caustics_refractive = False
        bpy.context.scene.cycles.use_auto_tile = False
        bpy.context.scene.render.resolution_x = img_w
        bpy.context.scene.render.resolution_y = img_h
        bpy.context.scene.render.resolution_percentage = 100
        bpy.context.scene.render.use_persistent_data = True
        self.resolution = np.array([img_w, img_h])
        bpy.context.scene.view_layers[0].cycles.use_denoising = True

        self.output_dir = Path("output")
        self._setup_compositor()
        self._setup_rendering_device()

    def get_new_object_id(self) -> int:
        self.__id_counter += 1
        return self.__id_counter

    def frame_set(self, frame_num: int):
        self._bl_scene.frame_set(frame_num)  # this sets the suffix for file names

    def set_output_path(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_node.base_path = str((self.output_dir).resolve())

    def set_mask_object(self, obj_id: int | None):
        if obj_id is None:
            self.output_node.file_slots[2].path = "mask/mask_"
            bpy.context.view_layer.use_pass_z = True
            bpy.context.view_layer.use_pass_combined = True
        else:
            self.output_node.file_slots[2].path = f"mask/mask_{obj_id:04d}_"
            bpy.context.view_layer.use_pass_z = False
            bpy.context.view_layer.use_pass_combined = False

    def get_cameras(self) -> List[Camera]:
        return [
            Camera(x) for x in self._bl_scene.collection.children["Cameras"].objects
        ]

    def create_camera(self, cam_name: str) -> Camera:
        cam = Camera.create(cam_name)
        # add camera to "Cameras" collection
        bpy.data.collections["Cameras"].objects.link(cam._bl_object)
        # remove from default collection
        bpy.context.scene.collection.objects.unlink(cam._bl_object)
        return cam

    def get_objects(self) -> List[Object]:
        return list(
            [Object(x) for x in self._bl_scene.collection.children["Objects"].objects]
        )

    def create_from_obj(
        self,
        obj_path: Path,
        add_physics: bool = False,
        mass: float = 1,
        friction: float = 0.5,
        restitution: float = 0.5,
    ) -> Object:
        obj = Object.from_obj(obj_path, add_physics, mass, friction, restitution)

        obj._bl_object.pass_index = self.get_new_object_id()
        # move to "Objects" collection
        bpy.context.scene.collection.objects.unlink(obj._bl_object)
        bpy.data.collections["Objects"].objects.link(obj._bl_object)
        return obj

    def create_copy(self, object: Object, linked: bool = False) -> Object:
        # clear blender selection
        bpy.ops.object.select_all(action="DESELECT")
        # select object
        object._bl_object.select_set(True)
        # returns a new object with a linked data block
        with redirect_stdout():
            bpy.ops.object.duplicate(linked=linked)
        bl_object = bpy.context.selected_objects[0]
        bl_object.pass_index = self.get_new_object_id()
        return Object(bl_object)

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

        logging.info(f"Available devices: {available_devices}")

        if chosen_type == "OPTIX":
            bpy.context.scene.cycles.denoiser = "OPTIX"
        else:
            bpy.context.scene.cycles.denoiser = "OPENIMAGEDENOISE"

    def set_background(self, filepath):
        # get composition node_tree
        img = bpy.data.images.load(filepath)
        self.bg_image_node.image = img
        scale_to_fit = np.max(self.resolution / np.array(img.size))
        self.bg_transform.inputs[4].default_value = scale_to_fit
        logging.info(f"Set background to {filepath}")

    def export_blend(self, filepath):
        with redirect_stdout():
            bpy.ops.wm.save_as_mainfile(filepath=filepath)

    def _setup_compositor(self):
        bpy.context.scene.use_nodes = True
        bpy.context.scene.render.film_transparent = True
        bpy.context.view_layer.use_pass_z = True
        bpy.context.view_layer.use_pass_object_index = True
        bpy.context.view_layer.use_pass_combined = True
        bpy.context.scene.render.use_multiview = True
        tree = bpy.context.scene.node_tree

        # clear node tree
        for node in tree.nodes:
            tree.nodes.remove(node)

        # add render layers node
        render_layers = tree.nodes.new("CompositorNodeRLayers")

        # create a alpha node to overlay the rendered image over the background image
        alpha_over = tree.nodes.new("CompositorNodeAlphaOver")
        self.bg_image_node = bg_image_node = tree.nodes.new("CompositorNodeImage")

        # create a transform node to scale the background image to fit the render resolution
        self.bg_transform = transform = tree.nodes.new("CompositorNodeTransform")
        transform.filter_type = "BILINEAR"

        # invert alpha
        invert_alpha = tree.nodes.new("CompositorNodeMath")
        invert_alpha.operation = "SUBTRACT"
        invert_alpha.inputs[0].default_value = 1.0

        # bg_node -> transform
        tree.links.new(bg_image_node.outputs[0], transform.inputs[0])
        # transform -> alpha over
        tree.links.new(transform.outputs[0], alpha_over.inputs[2])
        # rendered_rgb -> alpha over
        tree.links.new(render_layers.outputs[0], alpha_over.inputs[1])
        # rendered_alpha -> invert alpha
        tree.links.new(render_layers.outputs[1], invert_alpha.inputs[1])
        # invert alpha -> alpha_over
        tree.links.new(invert_alpha.outputs[0], alpha_over.inputs[0])

        # RGB image output
        self.output_node = output = tree.nodes.new("CompositorNodeOutputFile")
        output.base_path = str((self.output_dir).resolve())
        output.inputs.remove(output.inputs[0])

        # RGB output
        output.file_slots.new("rgb")
        output.file_slots[0].path = "rgb/rgb_"
        output.file_slots[0].use_node_format = False
        output.file_slots[0].format.color_mode = "RGB"
        output.file_slots[0].format.file_format = "PNG"
        tree.links.new(alpha_over.outputs[0], output.inputs["rgb"])

        # Depth output
        output.file_slots.new("depth")
        output.file_slots[1].path = "depth/depth_"
        output.file_slots[1].use_node_format = False
        output.file_slots[1].format.color_mode = "RGB"
        output.file_slots[1].format.file_format = "OPEN_EXR"
        output.file_slots[1].format.exr_codec = "ZIP"
        output.file_slots[1].format.color_depth = "16"
        tree.links.new(render_layers.outputs["Depth"], output.inputs["depth"])

        # Object index output
        ret_val = output.file_slots.new("object_index")
        output.file_slots[2].path = "mask/mask_"
        output.file_slots[2].use_node_format = False
        output.file_slots[2].format.color_mode = "RGB"
        output.file_slots[2].format.file_format = "OPEN_EXR"
        output.file_slots[2].format.exr_codec = "ZIPS"  # lossless
        output.file_slots[2].format.color_depth = "16"
        tree.links.new(render_layers.outputs["IndexOB"], output.inputs["object_index"])
