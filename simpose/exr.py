import OpenEXR as exr
import Imath
from pathlib import Path
import numpy as np


class EXR:
    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath

    def read(self, channel_name: str) -> np.ndarray:
        if not self.filepath.exists():
            raise FileNotFoundError(f"File {self.filepath} does not exist")

        exr_file = exr.InputFile(str(self.filepath))
        try:
            header = exr_file.header()
            channels = header["channels"]
            height, width = header["dataWindow"].max.y + 1, header["dataWindow"].max.x + 1

            exr_channel = channels[channel_name]
            pixel_type = exr_channel.type
            np_dtype: type[np.floating | np.integer]
            if pixel_type == Imath.PixelType(Imath.PixelType.FLOAT):
                np_dtype = np.float32
            elif pixel_type == Imath.PixelType(Imath.PixelType.UINT):
                np_dtype = np.uint32
            elif pixel_type == Imath.PixelType(Imath.PixelType.HALF):
                np_dtype = np.float16
            else:
                raise ValueError(f"Unsupported EXR pixel type: {pixel_type}")

            img = np.frombuffer(
                exr_file.channel(
                    channel_name,
                    pixel_type,
                ),
                dtype=np_dtype,
            ).reshape((height, width))
            return img
        finally:
            exr_file.close()

    def write(
        self,
        channels: dict[str, np.ndarray],
    ) -> None:
        """write dict of Channels to a multilayer EXR file. Only np arrays with np.float16, np.float32 or np.uint32 are supported"""
        if len(channels) == 0:
            raise ValueError("No channels to write")

        img = list(channels.values())[0]
        header = exr.Header(
            img.shape[1],
            img.shape[0],
        )
        header["compression"] = Imath.Compression(Imath.Compression.NO_COMPRESSION)

        if img.dtype == np.float32:
            exr_type = Imath.PixelType(Imath.PixelType.FLOAT)
        elif img.dtype == np.uint32:
            exr_type = Imath.PixelType(Imath.PixelType.UINT)
        elif img.dtype == np.float16:
            exr_type = Imath.PixelType(Imath.PixelType.HALF)
        else:
            raise ValueError(f"Unsupported dtype do write to EXR: {img.dtype}")

        header["channels"] = dict({k: Imath.Channel(exr_type) for k in channels.keys()})

        exr_file = exr.OutputFile(str(self.filepath), header)
        try:
            exr_file.writePixels({k: img.tobytes() for k, img in channels.items()})
        finally:
            exr_file.close()
