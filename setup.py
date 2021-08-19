import glob

import setuptools
from setuptools import setup

setup(
    name="k8s_DP",
    version="1.1",
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
        "SimpleITK",
        "pandas",
        "scikit-learn",
        "tqdm",
        "MedPy",
        "plotly",
        "kaleido",
        "nbformat",
        "pandasgui",
    ],
    scripts=glob.glob("scripts/*"),
)
