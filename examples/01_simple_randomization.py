import simpose as s
from pathlib import Path

scene = s.Scene()

# use context manager to explicitly specify the blender scene, where the objects are created
with scene:
    duck = s.Object.from_obj(str(Path("../meshes/cpsduck/cpsduck.obj").resolve()), object_id=1)
    cam = s.Camera("Camera")


# export Scene as .blend file, so we can open it in Blender and check results
scene.export_blend(str(Path("scene.blend").resolve()))

for i in range(10):
    s.random.randomize_rotation(duck)
    s.random.randomize_in_camera_frustum(duck, cam, (1.0, 2.0), (0.9, 0.9))
    
    scene.render()