import pathlib
import re

with pathlib.Path("log.txt").open("r") as F:
    lines = F.readlines()

lines = [l.strip() for l in lines if "semaphore" in l.lower()]

semaphore_id_regex = r"(\(<threading\.BoundedSemaphore object at 0x[a-z0-9]+>\))"

progress_regex = r"\d%\|.+\| \d+/\d+ \[.+\]"

semaphore_names = {}
sem_values = {}
for i, line in enumerate(lines):
    match_result = re.search(semaphore_id_regex, line)
    if match_result:
        semaphore_id = match_result.group(1)
        if semaphore_id not in semaphore_names:
            semaphore_names[semaphore_id] = f"SEM{len(semaphore_names)}"
            sem_values[semaphore_id] = 6
        # line = line.replace(semaphore_id, semaphore_names[semaphore_id])
        lines[i] = re.sub(semaphore_id_regex, semaphore_names[semaphore_id], line)
        lines[i] = re.sub(progress_regex, "", lines[i])
        lines[i] = re.sub("DEBUG", "", lines[i])
        lines[i] = re.sub("GPU", "", lines[i])
        if "acquired" in line.lower():
            sem_values[semaphore_id] -= 1
        elif "released" in line.lower():
            sem_values[semaphore_id] += 1
        lines[i] += f" value: {sem_values[semaphore_id]}"

#lines = [l for l in lines if "sem1" in l.lower()]

with pathlib.Path("out.txt").open("w") as F:
    F.write("\n".join(lines))
