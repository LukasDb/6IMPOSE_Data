import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import bpy

BL_OPS: list["type[bpy.types.Operator]"] = []
import logging, coloredlogs  # type: ignore
import multiprocessing as mp

logger = mp.get_logger()

coloredlogs.install(
    fmt=f"%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    logger=logger,
    reconfigure=False,
)

from simpose.render_product import RenderProduct, ObjectAnnotation
from simpose.exr import EXR
from simpose.remote_semaphore import RemoteSemaphore
from simpose.redirect_stdout import redirect_stdout
import simpose.observers as observers
import simpose.entities as entities
from simpose.scene import Scene
import simpose.random as random
import simpose.writers as writers
import simpose.generators as generators
import simpose.downloaders as downloaders
import simpose.data as data
