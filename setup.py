from setuptools import setup, find_packages

setup(
    name="simpose",
    version="0.0.1",
    description="Synthetic Image Generator based on Blender",
    url="todo",
    author="Lukas Dirnberger",
    author_email="lukas.dirnberger@tum.de",
    license="todo",
    packages=find_packages(),
    install_requires=open("requirements.txt").readlines(),
    entry_points= {"console_scripts": ["simpose = simpose.__main__:run"]},
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.10",
    ],
)