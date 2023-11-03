from abc import ABC, abstractmethod
from pathlib import Path


class Downloader(ABC):
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    @abstractmethod
    def run(self):
        pass
