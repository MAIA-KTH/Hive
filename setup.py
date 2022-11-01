import os

import setuptools
from setuptools import setup


def resolve_requirements(file):
    requirements = []
    with open(file) as f:
        req = f.read().splitlines()
        for r in req:
            if r.startswith("-r"):
                requirements += resolve_requirements(os.path.join(os.path.dirname(file), r.split(" ")[1]))
            else:
                requirements.append(r)
    return requirements


def read_file(file):
    with open(file) as f:
        content = f.read()
    return content


setup(
    name="Hive",
    version="1.0",
    url="https://github.com/MAIA-KTH/Hive.git",
    license="GPLv3",
    author="Simone Bendazzoli",
    author_email="simben@kth.se",
    description="Python Package to support Deep Learning data preparation, pre-processing. training, result visualization"
    " and model deployment across different frameworks (nnUNet, nnDetection, MONAI).",
    long_description=read_file(os.path.join(os.path.dirname(__file__), "README.md")),
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    package_data={
        "": ["configs/*.yml", "configs/*.json"],
    },
    # package_dir={"": "src"},
    install_requires=resolve_requirements(os.path.join(os.path.dirname(__file__), "requirements.txt")),
    entry_points={
        "console_scripts": [
            "Hive_convert_DICOM_dataset_to_NIFTI_dataset = scripts.Hive_convert_DICOM_dataset_to_NIFTI_dataset:main",
            "Hive_run_pipeline_from_file = scripts.Hive_run_pipeline_from_file:main",
            "Hive_create_subset = scripts.Hive_create_subset:main",
            "nndet_create_pipeline = scripts.nndet_create_pipeline:main",
            "nndet_prepare_data_folder = scripts.nndet_prepare_data_folder:main",
            "nndet_run_preprocessing = scripts.nndet_run_preprocessing:main",
            "nndet_run_training = scripts.nndet_run_training:main",
            "Hive_convert_semantic_to_instance_segmentation = scripts.Hive_convert_semantic_to_instance_segmentation:main",
        ],
    },
    keywords=["deep learning", "image segmentation", "medical image analysis", "medical image segmentation"],
    # scripts=glob.glob("scripts/*"),
)
