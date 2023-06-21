from typing import Dict
from simpose.camera import Camera
from simpose.light import Light
from simpose.object import Object
import bpy
import numpy as np
import sys
import io
import os
import logging
from .redirect_stdout import redirect_stdout


class Scene:
    def __init__(self) -> None:
        self._bl_scene = bpy.data.scenes.new("6impose Scene")
        bpy.context.window.scene = self._bl_scene

        # setup settings
        bpy.context.scene.render.engine = "CYCLES"
        # bpy.context.scene.cycles.device = 'GPU'
        bpy.context.scene.cycles.use_denoising = True
        # bpy.context.scene.cycles.denoiser = 'OPTIX'
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
        self.Tree = None
        self.rgbnode = None
        self.depthnode = None
        self.segmentNode = None
        self._setup_compositor()

    def render(self, i):
        self.segmentNode.file_slots[0].path = f"segment_{i}_"
        self.rgbnode.file_slots[0].path = f"rgb_{i}_"
        self.depthnode.file_slots[0].path = f"depth_{i}_"
        bpy.context.scene.render.filepath = f"render/rgb_images/render_{i}.png"
        with redirect_stdout():
            bpy.ops.render.render(write_still=True)

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
        tree = bpy.context.scene.node_tree
        self.Tree = tree
        bpy.context.view_layer.use_pass_z = True
        

        bpy.context.view_layer.use_pass_object_index = True

        bpy.context.view_layer.use_pass_combined = True

        # create a alpha node to overlay the rendered image over the background image
        alpha_over = tree.nodes.new("CompositorNodeAlphaOver")
        self.bg_image_node = bg_image_node = tree.nodes.new("CompositorNodeImage")

        # create a transform node to scale the background image to fit the render resolution
        self.bg_transform = transform = tree.nodes.new("CompositorNodeTransform")
        transform.filter_type = "BILINEAR"
        
    
        # link the nodes
        tree.links.new(bg_image_node.outputs[0], transform.inputs[0])
        tree.links.new(transform.outputs[0], alpha_over.inputs[1])
        tree.links.new(tree.nodes["Render Layers"].outputs[0], alpha_over.inputs[2])
        tree.links.new(alpha_over.outputs[0], tree.nodes["Composite"].inputs[0])
        tree.links.new(tree.nodes["Render Layers"].outputs[1], alpha_over.inputs[0])

        # RGB image output
        rgb_output_node = tree.nodes.new("CompositorNodeOutputFile")
        rgb_output_node.base_path = "render/rgb/"
        rgb_output_node.format.file_format = "PNG"
        self.rgbnode = rgb_output_node
        tree.links.new(self.Tree.nodes["Render Layers"].outputs["Image"], rgb_output_node.inputs["Image"])
        
        # Depth image output
        depth_output_node = tree.nodes.new("CompositorNodeOutputFile")
        depth_output_node.base_path = "render/depth_images/"
        depth_output_node.format.file_format = "PNG"
        self.depthnode = depth_output_node
        tree.links.new(self.Tree.nodes["Render Layers"].outputs["Depth"], depth_output_node.inputs["Image"])
        
        #id_mask_node = tree.nodes.new('CompositorNodeIDMask')
        segmentation_output_node = tree.nodes.new('CompositorNodeOutputFile')
        segmentation_output_node.base_path = "render/segmentation_images/"
        segmentation_output_node.format.file_format = "OPEN_EXR"
        self.segmentNode = segmentation_output_node
        
        tree.links.new(self.Tree.nodes["Render Layers"].outputs['IndexOB'], segmentation_output_node.inputs['Image'])
        #tree.links.new(id_mask_node.outputs['IndexMA'], segmentation_output_node.inputs['Image'])

       

            
        
        


    




        






       



        



