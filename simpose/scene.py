from typing import Dict
from simpose.camera import Camera
from simpose.light import Light
from simpose.object import Object
import bpy
import numpy as np
import sys
import io
import os
from pathlib import Path
import json,time
import logging
from .redirect_stdout import redirect_stdout
import mathutils
from typing import List

class Scene:
    def __init__(self) -> None:
        self._bl_scene = bpy.data.scenes.new("6impose Scene")
        bpy.context.window.scene = self._bl_scene

        self._bl_scene["id_counter"] = 1 # 0 is reserved for background

        # setup settings
        bpy.context.scene.render.engine = "CYCLES"
        # bpy.context.scene.cycles.device = 'GPU'
        bpy.context.scene.cycles.use_denoising = True
        #bpy.context.scene.cycles.denoiser = 'OPTIX'
        bpy.context.scene.cycles.samples = 64
        bpy.context.scene.cycles.caustics_reflective = False
        bpy.context.scene.cycles.caustics_refractive = False
        bpy.context.scene.cycles.use_auto_tile = False
        bpy.context.scene.render.resolution_x = 640
        bpy.context.scene.render.resolution_y = 480
        bpy.context.scene.render.resolution_percentage = 100
        bpy.context.scene.render.use_persistent_data = True
        self.resolution = np.array([640, 480])
        bpy.context.scene.view_layers[0].cycles.use_denoising = True

        self.rgbnode = None
        self.depthnode = None
        self.segmentNode = None
        self._setup_compositor()
        self._setup_rendering_device()
        
    def _setup_rendering_device(self):
        bpy.context.scene.cycles.device = 'GPU'
        pref = bpy.context.preferences.addons["cycles"].preferences
        pref.get_devices()

        for dev in pref.devices:
            dev.use = False


        device_types = list({x.type for x in pref.devices})
        priority_list = ['OPTIX', 'HIP', 'ONEAPI', 'CUDA']

        chosen_type = "NONE"

        for type in priority_list:
            if type in device_types:
                chosen_type = type
                break

        # Set GPU rendering mode to detected one
        pref.compute_device_type = chosen_type

        chosen_type_device = "CPU" if chosen_type == "NONE" else chosen_type
        available_devices = [x for x in pref.devices if x.type == chosen_type_device]

        selected_devices = [0] # TODO parametrize this
        for i, dev in enumerate(available_devices):
            if i in selected_devices:
                dev.use = True

        logging.info(f"Available devices: {available_devices}")

        if chosen_type == 'OPTIX':
            bpy.context.scene.cycles.denoiser = 'OPTIX'
        else:
            bpy.context.scene.cycles.denoiser = 'OPENIMAGEDENOISE'

    def render(self, i):
        bpy.context.scene.frame_set(i) # this sets the suffix for file names
        self.segmentNode.file_slots[0].path = f"segment_"
        self.rgbnode.file_slots[0].path = f"rgb_"
        self.depthnode.file_slots[0].path = f"depth_"
        with redirect_stdout():
            bpy.ops.render.render(write_still=False)

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

    def __enter__(self):
        bpy.context.window.scene = self._bl_scene
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def _setup_compositor(self):
        bpy.context.scene.use_nodes = True
        bpy.context.scene.render.film_transparent = True
        bpy.context.view_layer.use_pass_z = True
        bpy.context.view_layer.use_pass_object_index = True
        bpy.context.view_layer.use_pass_combined = True

        tree = bpy.context.scene.node_tree

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
        tree.links.new(tree.nodes["Render Layers"].outputs[0], alpha_over.inputs[1])
        # rendered_alpha -> invert alpha
        tree.links.new(tree.nodes["Render Layers"].outputs[1], invert_alpha.inputs[1])
        # invert alpha -> alpha_over
        tree.links.new(invert_alpha.outputs[0], alpha_over.inputs[0])



        # RGB image output
        rgb_output_node = tree.nodes.new("CompositorNodeOutputFile")
        rgb_output_node.base_path = "render/rgb/"
        rgb_output_node.format.file_format = "PNG"
        self.rgbnode = rgb_output_node
        tree.links.new(alpha_over.outputs[0], rgb_output_node.inputs["Image"])
        
        # Depth image output
        depth_output_node = tree.nodes.new("CompositorNodeOutputFile")
        depth_output_node.base_path = "render/depth/"
        depth_output_node.format.file_format = "OPEN_EXR"
        self.depthnode = depth_output_node
        tree.links.new(tree.nodes["Render Layers"].outputs["Depth"], depth_output_node.inputs["Image"])
        
        #id_mask_node = tree.nodes.new('CompositorNodeIDMask')
        segmentation_output_node = tree.nodes.new('CompositorNodeOutputFile')
        segmentation_output_node.base_path = "render/mask/"
        segmentation_output_node.format.file_format = "OPEN_EXR"
        self.segmentNode = segmentation_output_node
        
        tree.links.new(tree.nodes["Render Layers"].outputs['IndexOB'], segmentation_output_node.inputs['Image'])
        
          
    def generate_data(self, data_dir,objs: List[Object],cam,i):
        self.render(i)
        
        obj_list = [{
                'class': obj.get_class(),
                'object id': obj.object_id,
                'pos': list(obj.location),
                'rotation': list(obj.rotation.as_quat()),
            }
            for obj in objs]
        
        cam_pos = cam.location
        cam_rot = cam.rotation
        cam_matrix = cam.get_calibration_matrix_K_from_blender()
        
        meta_dict= {
            'cam_rotation': list(cam_rot.as_quat()),
            'cam_location':list(cam_pos),
            'cam_matrix': np.array(cam_matrix).tolist(),
            'objs': list(obj_list)
            }
        if not os.path.isdir(os.path.join(data_dir)):
            os.makedirs(os.path.join(data_dir))
        with open(os.path.join(data_dir, f"gt_{i:05}.json"), 'w') as F:
            json.dump(meta_dict, F, indent=2)

            
        
    
    

        
