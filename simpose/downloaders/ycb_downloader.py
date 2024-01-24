# Copyright 2015 Yale University - Grablab
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:\
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# modified for simpose by lukas.dirnberger@tum.de

import os
import json
from pathlib import Path
import requests
import tqdm
from typing import Any

from .downloader import Downloader


class YCBDownloader(Downloader):
    base_url = "http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/"

    def __init__(self, output_dir: Path) -> None:
        super().__init__(output_dir)

    def run(self) -> None:
        objects_url = self.base_url + "objects.json"

        objects = self.fetch_objects(objects_url)

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        bar = tqdm.tqdm(total=len(objects))

        for object in objects:
            url = self.tgz_url(self.base_url, object, "google_16k")
            filename = "{path}/{object}_{file_type}.tgz".format(
                path=self.output_dir, object=object, file_type="google_16k"
            )
            bar.set_postfix_str(f"{filename}")
            self.download_file(url, filename)
            self.extract_tgz(filename)
            bar.update(1)

    def fetch_objects(self, url: str) -> Any:
        response = requests.get(url)
        html = response.text
        objects = json.loads(html)
        return objects["objects"]

    def download_file(self, url: str, filename: str) -> None:
        with open(filename, "wb") as F:
            r = requests.get(url, allow_redirects=True)
            F.write(r.content)

    def tgz_url(self, base_url: str, object: str, type: str) -> str:
        if type in ["berkeley_rgbd", "berkeley_rgb_highres"]:
            return base_url + "berkeley/{object}/{object}_{type}.tgz".format(
                object=object, type=type
            )
        elif type in ["berkeley_processed"]:
            return base_url + f"berkeley/{object}/{object}_berkeley_meshes.tgz"
        else:
            return base_url + "google/{object}_{type}.tgz".format(object=object, type=type)

    def extract_tgz(self, filename: str) -> None:
        if not Path(filename).exists():
            return
        # print(f"Extracting {filename}")
        tar_command = "tar -xzf {filename} -C {dir}".format(filename=filename, dir=self.output_dir)
        os.system(tar_command)
        os.remove(filename)
