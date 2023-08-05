import os
import coloredlogs, logging

coloredlogs.install(fmt="%(asctime)s %(levelname)s %(message)s", level="INFO")
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from simpose.callback import CallbackType, Callback, Callbacks
from simpose.scene import Scene
from simpose.camera import Camera
from simpose.object import Object
import simpose.random as random
from simpose.writer import Writer


__all__ = ["Scene", "Writer", "random", "Camera", "Object", "CallbackType"]
