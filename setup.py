from setuptools import setup, find_packages

package_list = [
    "tqdm",
    "bpy==3.6",
    "fake-bpy-module-latest",
    "scipy",
    "click",
    "pybullet",
    "coloredlogs",
    "streamlit",
    "openexr",
    "pydantic",
    "pyyaml",
    "tensorflow"
]

# check if opencv-contrib-python is installed
try:
    import cv2.aruco  # noqa: F401
except ImportError:
    package_list += ["opencv-python"]


setup(
    name="simpose",
    version="1.2.0",
    description="Synthetic Image Generator based on Blender",
    url="https://github.com/LukasDb/6IMPOSE_Data",
    author="Lukas Dirnberger",
    author_email="lukas.dirnberger@tum.de",
    license="MIT",
    packages=find_packages(),
    install_requires=package_list,
    entry_points={"console_scripts": ["simpose = simpose.__main__:run"]},
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.10",
    ],
)
