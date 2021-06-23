import glob

import setuptools
from setuptools import setup

setup(
    name="k8s_data_preparation",
    version="1.0",
    url="https://github.com/SimoneBendazzoli93/k8s_nnUNet.git",
    license="",
    author="Simone Bendazzoli",
    author_email="simben@kth.se",
    description="Tool to standardize original dataset folder to match Decathlon-like folder structure",  # noqa: E501
    packages=setuptools.find_packages("src"),
    package_data={
        "": ["*.yml", "*.json"],
    },
    package_dir={"": "src"},
    install_requires=[
        "coloredlogs",
        "matplotlib",
        "numpy",
        "SimpleITK",
        "seg-metrics",
        "pandas",
        "PySimpleGUI",
        "pynrrd",
        "nibabel",
        "scikit-image",
        "ipython",
    ],
    scripts=glob.glob("scripts/*"),
)
