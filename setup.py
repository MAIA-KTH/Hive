import os

import setuptools
from setuptools import setup

with open(os.path.join('VERSION')) as version_file:
    version = version_file.read().strip()


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
    name="hive-maia",
    version=version,
    url="https://github.com/MAIA-KTH/Hive.git",
    license="GPLv3",
    project_urls={
        "Documentation": "https://hive-maia.readthedocs.io",
        "Source": "https://github.com/MAIA-KTH/Hive",
        "Tracker": "https://github.com/MAIA-KTH/Hive/issues",
    },
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
    data_files=[('', ['VERSION', "requirements.txt"]), ],
    # package_dir={"": "src"},
    install_requires=resolve_requirements(os.path.join(os.path.dirname(__file__), "requirements.txt")),
    entry_points={
        "console_scripts": [
            "Hive_convert_DICOM_dataset_to_NIFTI_dataset = Hive_scripts.Hive_convert_DICOM_dataset_to_NIFTI_dataset:main",
            "Hive_run_pipeline_from_file = Hive_scripts.Hive_run_pipeline_from_file:main",
            "Hive_create_subset = Hive_scripts.Hive_create_subset:main",
            "nndet_create_pipeline = Hive_scripts.nndet_create_pipeline:main",
            "nndet_prepare_data_folder = Hive_scripts.nndet_prepare_data_folder:main",
            "nndet_run_preprocessing = Hive_scripts.nndet_run_preprocessing:main",
            "nndet_run_training = Hive_scripts.nndet_run_training:main",
            "Hive_convert_semantic_to_instance_segmentation = Hive_scripts.Hive_convert_semantic_to_instance_segmentation:main",
            "Hive_extract_experiment_predictions = Hive_scripts.Hive_extract_experiment_predictions:main",
            "nndet_compute_metric_results = Hive_scripts.nndet_compute_metric_results:main",
            "Hive_order_data_folder = Hive_scripts.Hive_order_data_folder:main",
        ],
    },
    keywords=["deep learning", "image segmentation", "medical image analysis", "medical image segmentation",
              "object detection"],
    # scripts=glob.glob("scripts/*"),
)
