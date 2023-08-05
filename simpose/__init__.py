from simpose.callback import CallbackType, Callback, Callbacks
from simpose.scene import Scene
from simpose.camera import Camera
from simpose.object import Object
import simpose.random as random
from simpose.writer import Writer

import logging
import coloredlogs

coloredlogs.install(fmt="%(asctime)s %(levelname)s %(message)s", level="INFO")

logging.info("Set logging format")


import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

__all__ = ["Scene", "Writer", "random", "Camera", "Object", "CallbackType"]
