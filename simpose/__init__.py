import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

BL_OPS = []
import logging, coloredlogs, multiprocessing

logger = multiprocessing.get_logger()

coloredlogs.install(
    level=logging.INFO,
    fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    logger=logger,
    reconfigure=False,
)

# tools
from simpose.redirect_stdout import redirect_stdout
from simpose.observers import Event

# 6impose
from simpose.entities import Object, Camera, Plane, Light
from simpose.scene import Scene
import simpose.random as random
import simpose.writers as writers
import simpose.generators as generators


__all__ = [
    "redirect_stdout",
    "Scene",
    "writers",
    "random",
    "generators",
    "Camera",
    "Object",
    "Plane",
    "Light",
    "Event",
    "logger",
]
