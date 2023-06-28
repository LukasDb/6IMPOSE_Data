import bpy
import json
import simpose
import numpy as np
from pathlib import Path
from .redirect_stdout import redirect_stdout


class Writer:
    def __init__(self, scene: simpose.Scene,output_dir: Path):
        self._output_dir = output_dir
        self._data_dir = output_dir / "data"
        self._scene = scene
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._scene.set_output_path(self._output_dir)

    
    """ contains logic to write the dataset """
    def generate_data(self, dataset_index: int):
        self._scene.frame_set(dataset_index) # this sets the suffix for file names
        
        # write rgb, depth, etc
        with redirect_stdout():
            bpy.ops.render.render(write_still=False)

        objs = self._scene.get_objects()

        # write metadata, GT annotations
        obj_list = [{
                'class': obj.get_class(),
                'object id': obj.object_id,
                'pos': list(obj.location),
                'rotation': list(obj.rotation.as_quat()),
            }
            for obj in objs]
        
        cam = self._scene.get_cameras()[0] # TODO in the future save all cameras
        cam_pos = cam.location
        cam_rot = cam.rotation
        cam_matrix = cam.get_calibration_matrix_K_from_blender()
        
        meta_dict= {
            'cam_rotation': list(cam_rot.as_quat()),
            'cam_location':list(cam_pos),
            'cam_matrix': np.array(cam_matrix).tolist(),
            'objs': list(obj_list)
            }
    
        with (self._data_dir/f"gt_{dataset_index:05}.json").open('w') as F:
            json.dump(meta_dict, F, indent=2)



        
