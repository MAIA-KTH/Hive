import glob

import setuptools
from setuptools import setup

setup(
    name="nnUNet_data_preparation",
    version="1.0",
    url="https://github.com/SimoneBendazzoli93/k8s_nnUNet.git",
    license="",
    author="Simone Bendazzoli",
    author_email="simben@kth.se",
    description="Tool to standardize dataset folder to match nnUNet folder structure",
    packages=setuptools.find_packages("src"),
    package_data={
        "": ["*.yml"],
    },
    package_dir={"": "src"},
    install_requires=[
        "setuptools",
        "coloredlogs",
        "argparse",
    ],
    scripts=glob.glob("scripts/*"),
)
