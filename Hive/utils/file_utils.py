import json
import os
import random
import shutil
from multiprocessing import Pool
from os import PathLike
from pathlib import Path
from typing import Union, List, Tuple, Dict

import SimpleITK
import nibabel as nib
import numpy as np
from tqdm import tqdm

from Hive.utils.log_utils import get_logger, DEBUG

logger = get_logger(__name__)


def subfiles(
    folder: Union[str, PathLike], join: bool = True, prefix: str = None, suffix: str = None, sort: bool = True
) -> List[str]:
    """
    Given a folder path, returns a list with all the files in the folder.

    Parameters
    ----------
    folder : Union[str, PathLike]
        Folder path.
    join : bool
        Flag to return the complete file paths or only the relative file names.
    prefix : str
        Filter the files with the specified prefix.
    suffix : str
        Filter the files with the specified suffix.
    sort : bool
        Flag to sort the files in the list by alphabetical order.

    Returns
    -------
    List[str] :
        Filename list.
    """
    if join:
        l = os.path.join  # noqa: E741
    else:
        l = lambda x, y: y  # noqa: E741, E731
    res = [
        l(folder, i.name)
        for i in Path(folder).iterdir()
        if i.is_file() and (prefix is None or i.name.startswith(prefix)) and (suffix is None or i.name.endswith(suffix))
    ]
    if sort:
        res.sort()
    return res


def subfolders(folder: Union[str, PathLike], join: bool = True, sort: bool = True) -> List[str]:
    """
     Given a folder path, returns a list with all the subfolders in the folder.

    Parameters
    ----------
    folder : Union[str, PathLike]
        Folder path.
    join : bool
        Flag to return the complete folder paths or only the relative folder names.
    sort : bool
        Flag to sort the sub folders in the list by alphabetical order.

    Returns
    -------
    List[str] :
        Sub folder list.
    """
    if join:
        l = os.path.join  # noqa: E741
    else:
        l = lambda x, y: y  # noqa: E741, E731
    res = [l(folder, i.name) for i in Path(folder).iterdir() if i.is_dir()]
    if sort:
        res.sort()
    return res


def create_nndet_data_folder_tree(data_folder: Union[str, PathLike], task_name: str, task_id: str):
    """
    Create nnDetection folder tree, ready to be populated with the dataset.

    nnDetection folder tree:

        ${raw_data_base}

        [Task000_Example]
            - dataset.yaml # dataset.json works too
            [raw_splitted]
                [imagesTr]
                    - case0000_0000.nii.gz # case0000 modality 0
                    - case0000_0001.nii.gz # case0000 modality 1
                    - case0001_0000.nii.gz # case0001 modality 0
                    - case0000_0001.nii.gz # case0001 modality 1
                [labelsTr]
                    - case0000.nii.gz # instance segmentation case0000
                    - case0000.json # properties of case0000
                    - case0001.nii.gz # instance segmentation case0001
                    - case0001.json # properties of case0001
                [imagesTs] # optional, same structure as imagesTr
                ...
                [labelsTs] # optional, same structure as labelsTr
                ...
            [preprocessed]
            [results]
        [Task001_Example1]
            ...


    Parameters
    ----------
    data_folder : Union[str, PathLike]
        folder path corresponding to the *raw_data_base* environment variable.
    task_name : str
        string used as task_name when creating task folder
    task_id : str
        string used as task_id when creating task folder
    """
    logger.log(DEBUG, ' Creating Dataset tree at "{}"'.format(data_folder))

    Path(data_folder).joinpath("Task" + task_id + "_" + task_name, "raw_splitted", "imagesTr",).mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(data_folder).joinpath("Task" + task_id + "_" + task_name, "raw_splitted", "labelsTr",).mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(data_folder).joinpath("Task" + task_id + "_" + task_name, "raw_splitted", "imagesTs",).mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(data_folder).joinpath("Task" + task_id + "_" + task_name, "raw_splitted", "labelsTs",).mkdir(
        parents=True,
        exist_ok=True,
    )


def split_dataset(input_data_folder: Union[str, PathLike], test_split_ratio: int, seed: int) -> Tuple[List[str], List[str]]:
    """
    Split dataset into a train/test split, given the specified ratio.

    Parameters
    ----------
    input_data_folder : Union[str, PathLike]
        folder path of the input dataset.
    test_split_ratio : int
        integer value in the range 0-100, specifying the split ratio to be used for the test set.
    seed : int
        integer value to be used as random seed.

    Returns
    -------
    Tuple[List[str], List[str]] :
        lists of strings containing subject IDs for train set and test set respectively.
    """
    subjects = subfolders(input_data_folder, join=False)

    random.seed(seed)
    random.shuffle(subjects)

    split_index = len(subjects) - int(len(subjects) * test_split_ratio / 100)

    train_subjects = subjects[0:split_index]
    test_subjects = subjects[split_index:]

    return train_subjects, test_subjects


def copy_image_file(input_filepath: Union[str, PathLike], output_filepath: Union[str, PathLike]):
    """
    Copy image file.

    Parameters
    ----------
    input_filepath : Union[str, PathLike]
        file path for the file to copy
    output_filepath : Union[str, PathLike]
        file path where to copy the file
    """
    shutil.copy(
        input_filepath,
        output_filepath,
    )


def copy_label_file(input_image: Union[str, PathLike], input_label: Union[str, PathLike], output_filepath: Union[str, PathLike]):
    """
    Copy label file, verifying the image information (spacing, orientation).

    Parameters
    ----------
    input_image : Union[str, PathLike]
        file path for the input image, to be used as reference when copying image information
    input_label : Union[str, PathLike]
        file path for the input label to be copied
    output_filepath : Union[str, PathLike]
        file location where to save the label image
    """
    label_nib = nib.load(input_label)
    image_nib = nib.load(input_image)

    label_nib_out = nib.Nifti1Image(label_nib.get_fdata(), image_nib.affine)
    nib.save(label_nib_out, output_filepath)


def copy_data_to_dataset_folder(
    input_data_folder: Union[str, PathLike],
    subjects: List[str],
    image_folder: Union[str, PathLike],
    config_dict: Dict[str, object],
    label_folder: Union[str, PathLike] = None,
    num_threads: int = None,
    save_label_instance_config: bool = False,
):
    """

    Parameters
    ----------

    input_data_folder : Union[str, PathLike]
        folder path of the input dataset
    subjects : List[str]
        string list containing subject IDs.
    image_folder : Union[str, PathLike]
        folder path where to store images (imagesTr/imagesTs).
    config_dict : Dict[str, object]
        dictionary with dataset and experiment configuration parameters.
    label_folder : Union[str, PathLike]
        folder path where to store labels (labelsTr/labelsTs). Default: ``None``.
        If **label_suffix** is ``None``, the label files are not saved.
    num_threads : int
        number of threads to use in multiprocessing ( Default: ``os.environ['N_THREADS']`` )
    save_label_instance_config : bool
        Flag to save label mask together with an instance dictionary as JSON file. NOTE: All the instances are assigned
         to instance class ``1``.
    """
    label_suffix = str(config_dict["label_suffix"])
    if num_threads is None:
        try:
            num_threads = int(os.environ["N_THREADS"])
        except KeyError:
            logger.warning("N_THREADS is not set as environment variable. Using Default [1]")
            num_threads = 1

    pool = Pool(num_threads)
    copied_files = []
    for directory in subjects:

        files = subfiles(
            str(Path(input_data_folder).joinpath(directory)),
            join=False,
            suffix=str(config_dict["FileExtension"]),
        )

        image_suffix_list = config_dict["Modalities"].keys()

        for modality, image_suffix in enumerate(image_suffix_list):
            modality_code = "_{0:04d}".format(modality)
            image_filename = directory + image_suffix

            if image_filename in files:
                updated_image_filename = image_filename.replace(image_suffix, modality_code + str(config_dict["FileExtension"]))
                copied_files.append(
                    pool.starmap_async(
                        copy_image_file,
                        (
                            (
                                str(Path(input_data_folder).joinpath(directory, image_filename)),
                                str(Path(image_folder).joinpath(updated_image_filename)),
                            ),
                        ),
                    )
                )
            else:
                logger.warning("{} is not found: skipping {} case".format(image_filename, directory))
        if label_suffix is not None and label_folder is not None and type(label_suffix) != list:

            label_filename = directory + label_suffix

            if label_filename in files:

                updated_label_filename = label_filename.replace(label_suffix, str(config_dict["FileExtension"]))

                copied_files.append(
                    pool.starmap_async(
                        copy_label_file,
                        (
                            (
                                str(Path(input_data_folder).joinpath(directory, directory + image_suffix)),
                                str(Path(input_data_folder).joinpath(directory, directory + label_suffix)),
                                str(Path(label_folder).joinpath(updated_label_filename)),
                            ),
                        ),
                    )
                )
                if save_label_instance_config:
                    label_map = SimpleITK.GetArrayFromImage(
                        SimpleITK.ReadImage(str(Path(input_data_folder).joinpath(directory, directory + label_suffix)))
                    )
                    instances = np.unique(label_map)
                    instances = instances[instances > 0]

                    json_dict = {
                        "instances": {str(int(i)): 0 for i in instances},
                    }
                    save_config_json(json_dict, str(Path(label_folder).joinpath(label_filename.replace(label_suffix, ".json"))))
            else:
                logger.warning("{} is not found: skipping {} case".format(label_filename, directory))

        elif type(label_suffix) == list and label_folder is not None:  # Multi Label
            for task_id, label_s in enumerate(label_suffix):
                task_code = "_{0:04d}".format(task_id)
                label_filename = directory + label_s

                if label_filename in files:

                    updated_label_filename = label_filename.replace(label_s, task_code + str(config_dict["FileExtension"]))

                    copied_files.append(
                        pool.starmap_async(
                            copy_label_file,
                            (
                                (
                                    str(Path(input_data_folder).joinpath(directory, directory + image_suffix)),
                                    str(Path(input_data_folder).joinpath(directory, directory + label_s)),
                                    str(Path(label_folder).joinpath(updated_label_filename)),
                                ),
                            ),
                        )
                    )
                else:
                    logger.warning("{} is not found: skipping {} case".format(label_filename, directory))

    _ = [i.get() for i in tqdm(copied_files)]


def save_config_json(config_dict: Dict[str, object], output_json: Union[str, PathLike]):
    """
    Save dictionary as JSON file.

    Parameters
    ----------
    output_json : Union[str, PathLike]
        JSON file path to be saved
    config_dict: Dict[str, object]
        dictionary to be saved in JSON format in the RESULTS_FOLDER
    """

    with open(output_json, "w") as fp:
        json.dump(config_dict, fp)


def generate_dataset_json(
    output_file: Union[str, PathLike],
    train_subjects: List[str],
    test_subjects: List[str],
    modalities: Tuple,
    labels: Union[Dict, List],
    dataset_name: str,
    task_name: str,
    n_tasks: int = 1,
):
    """
    Generates and saves a Dataset JSON file.

    Parameters
    ----------
    output_file : Union[str, PathLike]
        This needs to be the full path to the dataset.json you intend to write, so
    output_file='DATASET_PATH/dataset.json' where the folder DATASET_PATH points to is the one with the
    imagesTr and labelsTr subfolders.
    train_subjects : List[str]
        List of subjects in the train set.
    test_subjects : List[str]
        List of subjects in the test set.
    modalities : tuple
        tuple of strings with modality names. must be in the same order as the images (first entry
    corresponds to _0000.nii.gz, etc). Example: ('T1', 'T2', 'FLAIR').
    labels : Union[Dict, List]
        dict with int->str (key->value) mapping the label IDs to label names. Note that 0 is always
    supposed to be background! Example: {0: 'background', 1: 'edema', 2: 'enhancing tumor'}. In case of a multi label task,
        the dictionaries for each label task are nested into a list.
    dataset_name : str
        The name of the dataset.
    task_name : str
        The name of the dataset.
    n_tasks : int
        Number of tasks. Default: 1.
    """
    task_type = []
    if type(labels) == dict:
        if len(labels) > 0:
            task_type.append("CLASSIFICATION")
        else:
            task_type.append("REGRESSION")
    elif type(labels) == list:
        for label in labels:
            if len(label) > 0:
                task_type.append("CLASSIFICATION")
            else:
                task_type.append("REGRESSION")

    json_dict = {
        "name": dataset_name,
        "task": task_name,
        "dim": 3,
        "test_labels": True,
        "task_type": task_type,
        "tensorImageSize": "4D",
        "modalities": {str(i): modalities[i] for i in range(len(modalities))},
        "labels": labels,  # {str(i): labels[i] for i in labels.keys()},
        "numTraining": len(train_subjects),
        "numTest": len(test_subjects),
        "n_tasks": n_tasks,
        "training": [{"image": "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in train_subjects],
        "test": ["./imagesTs/%s.nii.gz" % i for i in test_subjects],
    }

    if not str(output_file).endswith("dataset.json"):
        print(
            "WARNING: output file name is not dataset.json! This may be intentional or not. You decide. "  # noqa: E501
            "Proceeding anyways..."
        )
    save_config_json(json_dict, output_file)
