# 6IMPOSE Data generation (v2)
Streamlined package of the 6IMPOSE data generation pipeline. Easier to install, use and to extend.

## Setup
If you encouter problems with OpenEXR try: `conda install -c conda-forge openexr-python` 
install this package with 
`pip install -e .`

## Usage
If you want to write your own dataset generation script, a typical script would follow this approach. Take a look at the examples!
- create a simpose Scene
- attach a Writer
- create a simpose Camera
- add objects, lights, etc...
- use the writer to generate the Datset
After you generated your dataset you can inspect the rendered data and labels with the Dataset Viewer:
```
simpose view <dataset_directory>
```

## Features
- Abstracted Blender interface
- Automatic rendering device setup
- Included Writer for writing the dataset to disk
- DatesetViewer
- GT labels with visible mask, mask without occlusions, bounding box and 6D pose
- Object import from .ply and .obj files
- Randomizer for Background images
- Randomizer for objects' poses
- Randomizer for lighting
- Randomizer using shapenet objects
- Physics using PyBullet
- Check out the examples!

## Models and Meshes 
- You can use the simpose.random.ModelLoader to retrieve random objects with .get_objects from the ShapeNet dataset. Please request and download the dataset from ShapeNet on your own and specify the path to the dataset in the ShapenetLoader.
- A script to download and extract Models from the YCB dataset is provided. The models can then be used in a similar fashion with the simpose.random.ModelLoader.
- The objects from [SynthDet](https://github.com/Unity-Technologies/SynthDet) can also be used as distractor objects. Download the "SynthDet/SynthDet/Assets/Foreground Objects" folder and convert the ASCII Fbx files to binary FBX files (for example, using [this](https://www.autodesk.com/developer-network/platform-technologies/fbx-sdk-2020-0?us_oa=akn-us&us_si=9066be5d-863a-4cd3-b98f-87bda034316b&us_st=fbx%20sdk))and specify the model source for the `simpose.random.ModelLoader`.

## Notes
- GT labels are only written for objects with `add_semantics=True`
- multiprocessing works, if you import simpose in the worker processes. Check memory usage!
- Currently implemented randomizers, specify the execution with `sp.CallbackType`:
    - BackgroundRandomizer: randomizes the background image
    - CameraFrustumRandomizer: randomizes object's pose in the camera view
    - LightRandomizer: Random number of point lights with random positions, colors and energy
    - ShapenetLoader: Loads random objects from the shapenet dataset. Should be used manually and not as a Callback. Download the [ShapeNet](https://shapenet.org) Core dataset and specify the path.
- If you want to use physics, add mass and friction to objects. The physics engine is PyBullet. The scene always has a collision plane at z=0 and gravity in negative z-direction. To progress the physics simulation use `scene.step_physics` with a timestep in seconds.


## Included Dataset Writer
The included dataset writer generates a dataset with instance annotations. If `render_object_masks=True`, the writer will render masks for each object with semantics without occlusions and calculate the visible fraction of pixels, as well as the bounding box without occlusions. This will slow down rendering, but includes additional information in the dataset. The writer will create a folder structure like this:

```
├── <output_path>/
|	├── depth/
|	|	├── depth_0000.exr 			-> depth from camera in meters
|	|	├── depth_0001.exr
|	|	├── ...
|	├── gt/
|	|	├── gt_0000.json 			-> explained below
|	|	├── gt_0001.json
|	|	├── ...
|	├── mask/
|	|	├── mask_0000.exr 	        -> instance IDs of visible objects
|	|	├── mask_<id>_0000.exr      -> if render_object_masks=True
|	|	├── mask_0001.exr
|	|	├── mask_<id>_0001.exr
|	|	├── ...
|	├── rgb/
|	|	├── rgb_0000.png 			-> RGB Image
|	|	├── rgb_0001.png
|	|	├── ...
```

For each datapoint, `gt_<datapoint_index>.json` contains the following information:

- `"cam_rotation"`: quaternion for the camera rotation as [x,y,z,w]
- `"cam_location"`: camera position as [x,y,z]
- `"cam_matrix"`: intrinsic camera matrix in OpenCV format
- `"objs"`: List of labels for all objects in this datapoint:
    - `"class"`: class name as a string, derived from the object filename
    - `"object id"`: instance id as `int`, unique for all objects, only valid for current datapoint
    - `"pos"`: object position in world frame [x,y,z]
    - `"rotation"`: quaternion in world frame [x,y,z,w]
    - `"bbox_visib"`: [x1,y1,x2,y2] in absolute pixels, bounding box of visible object
    - `"bbox_obj"`: [x1,y1,x2,y2] in absolute pixels, bounding box of the object without occlusions
    - `"px_count_visib"`: count of visible pixels for this object
    - `"px_count_valid"`: count of visible pixels with valid depth for this object
    - `"px_count_all"`: count of pixels without occlusions for this object
    - `"visib_fract"`: px_count_visib / px_count_all (or 0.)

If `render_object_masks = False`:
There will be no rendered masks without occlusions `masks/mask_<object id>_<datapoint_index>.exr` and `"px_count_all"` and `"visib_fract"` will be `0.0`, and `"bbox_obj"` will be `[0, 0, 0, 0]` for all objects.
