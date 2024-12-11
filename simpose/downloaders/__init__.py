from .downloader import Downloader
from .ycb_downloader import YCBDownloader

from .omni3d_downloader import Omni3DDownloader

__all__ = ["YCBDownloader", "Omni3DDownloader"]

__datasets__ = [
    "YCB",
    "Omni3D",
    "LineMOD",  # TODO
    "Hope",  # TODO
    "Homebrew",  # TODO
]
