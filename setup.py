import glob

import setuptools
from setuptools import setup

setup(
    name="Hive",
    version="2.0",
    url="https://github.com/SimoneBendazzoli93/Hive.git",
    license="",
    author="Simone Bendazzoli",
    author_email="simben@kth.se",
    description="",  # noqa: E501
    packages=setuptools.find_packages("src"),
    package_data={
        "": ["configs/*.yml", "configs/*.json"],
    },
    package_dir={"": "src"},
    install_requires=[
        "visdom",
        "coloredlogs",
        "numpy",
        "nptyping",
        "SimpleITK",
        "pandas",
        "scipy",
        "tqdm",
        "MedPy",
        "plotly",
        "kaleido",
        "nbformat",
        "pandasgui",
        "xlsxwriter",
        "openpyxl",
    ],
    extras_require={"Monai-env": ["monai", "pytorch-ignite"], "nnUNet-env": ["nnunet"]},
    scripts=glob.glob("scripts/*"),
)
