# 6IMPOSE Data generation (v2)
Streamlined package of the 6IMPOSE data generation pipeline. Easier to install, use and to extend and also used to create the *Dropjects* datasets (link to the datasets will follow). If you want to use 6IMPOSE_Data in your academic work, please cite the following 2 papers:
```
@ARTICLE{cao2023simpose,
    AUTHOR={Cao, Hongpeng and Dirnberger, Lukas and Bernardini, Daniele and Piazza, Cristina and Caccamo, Marco},   
    TITLE={6IMPOSE: bridging the reality gap in 6D pose estimation for robotic grasping},      
    JOURNAL={Frontiers in Robotics and AI},      
    VOLUME={10},
    YEAR={2023},
    URL={https://www.frontiersin.org/articles/10.3389/frobt.2023.1176492},       
    DOI={10.3389/frobt.2023.1176492},      
    ISSN={2296-9144},
}

TBA

```

## Setup
If you encouter problems with OpenEXR try: `conda install -c conda-forge openexr-python` 
install this package with 
`pip install -e .`

## Usage
### Preview
After you generated or downloaded a dataset, you can inspect the rendered data and labels with the Dataset Viewer. This works also with concatenating all datasets found in that directory.
```
simpose view <dataset_directory>
```
### Included Generators
6IMPOSE has built-in dataset generators, to quickly generate datasets without the need for custom scripts. To use a generator, run
```
simpose generate <config.yaml>
```
,where the config file specifies the parameters and the generator for dataset generation. You can generate a config file with `simpose generate -i <config.yaml>`, where you will be prompted to choose a Generator (e.g. DroppedObjects). The config file will be generated with default parameters. You can then edit the config file and run `simpose generate <config.yaml>` to generate the dataset. Make sure to specify all necessary file paths and check if the defaults are suitable for your use case!

### Custom Usage
If you want to write your own dataset generation script, a typical script would follow this approach. Take a look at the included generators!
- create a simpose Scene
- attach a Writer
- create a simpose Camera
- add objects, lights, etc...
- use the writer to generate the Datset


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
You can use the simpose.random.ModelLoader to load procedurally load random models. You can specify to load all .obj, .ply or .fbx files from the specified folder. You can also specify an `objects.json` with  a list of objects by their canonical name and path to the mesh file: `<name>: <path>`.

## Notes
- multiprocessing works, if you import simpose in the worker processes. Check memory usage!
- Currently implemented randomizers, specify the execution with `sp.CallbackType`:
    - BackgroundRandomizer: randomizes the background image
    - CameraFrustumRandomizer: randomizes object's pose in the camera view
    - CameraSceneRandomizer: randomizes camera pose around the origin with some constraints
    - LightRandomizer: Random number of point lights with random positions, colors and energy
    - ModelLoader: Loads random models from a specified folder
- If you want to use physics, add mass and friction to objects. The physics engine is PyBullet. The scene always has a collision plane at z=0 and gravity in negative z-direction. To progress the physics simulation use `scene.step_physics` with a timestep in seconds.


## Included Dataset Writer
The dataset generation will create sharded TFrecord files, compressed with ZLIB. The dataset will be written to the specified directory. Each datapoint, contains the following information:

- `"cam_rotation"`: quaternion for the camera rotation as [x,y,z,w]
- `"cam_location"`: camera position as [x,y,z]
- `"cam_matrix"`: intrinsic camera matrix in OpenCV format
- `"objs"`: List of labels for all objects in this datapoint:
    - `"class"`: class name as a string, derived from the object filename
    - `"obj_id"`: instance id as `int`, unique for all objects, only valid for current datapoint
    - `"pos"`: object position in world frame [x,y,z]
    - `"rotation"`: quaternion in world frame [x,y,z,w]
    - `"bbox_visib"`: [x1,y1,x2,y2] in absolute pixels, bounding box of visible object
    - `"bbox_obj"`: [x1,y1,x2,y2] in absolute pixels, bounding box of the object without occlusions
    - `"px_count_visib"`: count of visible pixels for this object
    - `"px_count_valid"`: count of visible pixels with valid depth for this object
    - `"px_count_all"`: count of pixels without occlusions for this object
    - `"visib_fract"`: px_count_visib / px_count_all (or 0.)
If an object is not visible the bounding box will be [0,0,00].


## Included Dataset Reader
> Note: `num_parallel_files` can have a huge impact on RAM and CPU usage. Recommended are much lower values or even start with `num_parallel_files=1`.

For easier use of the generated Dataset, 6IMPOSE_Data includes a high-performance Tensorflow dataloader. TFRecordDataset will search the root folder for a dataset. If multiple datasets are found, then the datasets will be concatenated. Usage:
```
import simpose as sp

tensorflow_dataset = sp.data.TFRecordDataset.get(
        root_dir: Path,
        get_keys: None | list[str] = None,
        pattern: str = "*.tfrecord",
        num_parallel_files: int = 1024
        )

# some example
for data in tensorflow_datasat:
    rgb = data[sp.data.Dataset.RGB]                 # [h,w,3] tf.uint8
    classes = data[sp.data.Dataset.OBJ_CLASSES]     # [n,] tf.string
    ids = data[sp.data.Dataset.OBJ_IDS]             # [n,] tf.int64
    mask = data[sp.data.Dataset.MASK]               # [h,w] tf.int64

    for cls, id in zip(classes, ids):
        if cls == 'some_class':
            # to get a binary mask for a specific object
            mask = tf.where(mask == id, 1, 0) # [h,w] tf.int64
            

```
The resulting Dataset is a regular `tf.data.Dataset` yielding dictionaries with the keys specified in get_keys or all of them if get_keys is not specified. The following keys according to the specification above are available:
```
simpose.data.Dataset.RGB
simpose.data.Dataset.RGB_R
simpose.data.Dataset.DEPTH
simpose.data.Dataset.DEPTH_R
simpose.data.Dataset.MASK
simpose.data.Dataset.GT
simpose.data.Dataset.CAM_MATRIX
simpose.data.Dataset.CAM_LOCATION
simpose.data.Dataset.CAM_ROTATION
simpose.data.Dataset.STEREO_BASELINE
simpose.data.Dataset.OBJ_CLASSES
simpose.data.Dataset.OBJ_IDS
simpose.data.Dataset.OBJ_POS
simpose.data.Dataset.OBJ_ROT
simpose.data.Dataset.OBJ_BBOX_VISIB
simpose.data.Dataset.OBJ_VISIB_FRACT
simpose.data.Dataset.OBJ_PX_COUNT_VISIB
simpose.data.Dataset.OBJ_PX_COUNT_VALID
simpose.data.Dataset.OBJ_PX_COUNT_ALL
simpose.data.Dataset.OBJ_BBOX_OBJ
```
Optionally available are the following keys:
```
simpose.data.Dataset.DEPTH_GT # obtained by any technique that can not be used at regular inference (such as multiple frames, etc...)
simpose.data.Dataset.DEPTH_GT_R # same as above
```

## Included Dataset Loaders
For easier access to existing datasets, 6IMPOSE_Data includes a high-performance Tensorflow dataloader. Usage is the same as you would use the TFRecordDataset, if no dataset is found in that directory, the loader will automatically start downloading and preprocessing the data. This might take a while and can potentially use a lot of disk space. Usage:
```
datasets = sp.data.LineMod.get(root_dir) # tf.data.Dataset
lm_dataset = sp.data.LineMod.get(root_dir) # full concatenated dataset
duck_dataset = sp.data.LineMod.get(root_dir.joinpath("subsets/000009")) # only a subset
duck_id = sp.data.LineMod.CLASSES['duck'] # 9
```
Use the same keys as above, but the available data depends on the original dataset. The datasets for the BOP Challenge are organized in subsets, usually for a single scene. These can be found separately in root_dir/subsets, each of which is a regular sp.data.TFRecordDataset. The models of the objects can also be found in the root_dir, according to the BOP format. The YCB-V dataset is the slightly modified version from the [BOP challenge](https://bop.felk.cvut.cz/datasets/) The following dataset are intended for local benchmarking, thus only the 'test' or 'validation' datasets are included, that contain full GT labels.:
- [sp.data.LineMod](http://campar.in.tum.de/Main/StefanHinterstoisser)
- [sp.data.LineModOccluded](https://heidata.uni-heidelberg.de/dataset.xhtml?persistentId=doi:10.11588/data/V4MUMX)
- [sp.data.TLess](http://cmp.felk.cvut.cz/t-less/)
- [sp.data.HomebrewedDB](http://campar.in.tum.de/personal/ilic/homebreweddb/index.html)
- [sp.data.YCBV](https://rse-lab.cs.washington.edu/projects/posecnn/)
- [sp.data.HOPE](https://github.com/swtyree/hope-dataset)
- WIP: COCO, ADE20k, TOD (transparent objects dataset)