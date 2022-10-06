import glob

import setuptools
from setuptools import setup

setup(
    name="Hive",
    version="2.1",
    url="https://github.com/SimoneBendazzoli93/Hive.git",
    license="GPLv3",
    author="Simone Bendazzoli",
    author_email="simben@kth.se",
    description="Python Package to support Deep Learning data preparation, pre-processing. training, result visualization"
                " and model deployment across different frameworks (nnUNet, nnDetection, MONAI).",
    packages=setuptools.find_packages("src"),
    package_data={
        "": ["configs/*.yml", "configs/*.json"],
    },
    package_dir={"": "src"},
    install_requires=["tqdm", "coloredlogs", "dicom2nifti", "nibabel", "scikit_learn", "numpy", "pydicom", "pynvml",
                      "SimpleITK"],
    scripts=glob.glob("scripts/*"),
)
