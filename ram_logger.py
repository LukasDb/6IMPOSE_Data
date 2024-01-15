import time
import os
import subprocess
import time
from pathlib import Path
import re

import gpu

file = Path("ram_log.txt")

file.unlink(missing_ok=True)

with file.open("w") as F:
    F.write(";".join(["Time (sec)", "Used RAM", "Used VRAM", "Temp GPU (°C)"]) + "\n")


t_start = time.time()

r_vram_usage = r"FB Memory Usage\n[ \w:]+\n[ \w:]+\n[ ]+Used[ ]+: (?P<used_vram>[0-9]+)"
r_gpu_tem = r"Temperature\n[ ]+GPU Current Temp[ ]+: (?P<gpu_temp>[0-9]+)"


while True:
    current = f"{time.time() - t_start:.2f}"
    ram_string = subprocess.run(["free", "-m"], capture_output=True, text=True).stdout
    used_ram = str(float(ram_string.splitlines()[1].split()[2]) / 1024)

    gpu_string = subprocess.run(["nvidia-smi", "-q"], capture_output=True, text=True).stdout

    used_vram_res = re.search(r_vram_usage, gpu_string)
    assert used_vram_res is not None
    used_vram = str(float(used_vram_res.group("used_vram")) / 1024)

    gpu_temp_res = re.search(r_gpu_tem, gpu_string)
    assert gpu_temp_res is not None
    gpu_temp = gpu_temp_res.group("gpu_temp")

    with file.open("a") as F:
        F.write(";".join([current, used_ram, used_vram, gpu_temp]) + "\n")

    time.sleep(1)
