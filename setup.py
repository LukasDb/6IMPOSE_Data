from setuptools import setup, find_packages

package_list = [
    "silence_tensorflow",
    "tqdm",
    "bpy",
    "scipy",
    "click",
    "pybullet",
    "coloredlogs",
    "streamlit",
    "streamlit-image-comparison",
    "openexr",
    "pydantic",
    "pyyaml",
    "tensorflow",
    "openxlab"
]

# check if opencv-contrib-python is installed
try:
    import cv2.aruco  # noqa: F401
except ImportError:
    package_list += ["opencv-python"]


setup(
    name="simpose",
    version="1.5.0",
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

print("Please install OpenEXR package yourself.")
