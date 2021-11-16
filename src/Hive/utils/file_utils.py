import itertools
import json
import os
import random
import shutil
from multiprocessing import Pool
from os import PathLike
from pathlib import Path
from typing import Dict, Tuple, List, Union

import nibabel as nib
import numpy as np
from tqdm import tqdm

from .log_utils import get_logger, DEBUG

logger = get_logger(__name__)


def subfiles(folder, join=True, prefix=None, suffix=None, sort=True):
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


def subfolders(folder, join=True, sort=True):
    if join:
        l = os.path.join  # noqa: E741
    else:
        l = lambda x, y: y  # noqa: E741, E731
    res = [l(folder, i.name) for i in Path(folder).iterdir() if i.is_dir()]
    if sort:
        res.sort()
    return res


def get_identifiers_from_splitted_files(folder: str):
    uniques = np.unique([i[:-12] for i in subfiles(folder, suffix=".nii.gz", join=False)])
    return uniques


def generate_dataset_json(
    output_file: str,
    imagesTr_dir: str,
    imagesTs_dir: str,
    modalities: Tuple,
    labels: dict,
    dataset_name: str,
    n_tasks: int = 1,
    license: str = "hands off!",
    dataset_description: str = "",
    dataset_reference="",
    dataset_release="0.0",
):
    """
    :param output_file: This needs to be the full path to the dataset.json you intend to write, so
    output_file='DATASET_PATH/dataset.json' where the folder DATASET_PATH points to is the one with the
    imagesTr and labelsTr subfolders
    :param imagesTr_dir: path to the imagesTr folder of that dataset
    :param imagesTs_dir: path to the imagesTs folder of that dataset. Can be None
    :param modalities: tuple of strings with modality names. must be in the same order as the images (first entry
    corresponds to _0000.nii.gz, etc). Example: ('T1', 'T2', 'FLAIR').
    :param labels: dict with int->str (key->value) mapping the label IDs to label names. Note that 0 is always
    supposed to be background! Example: {0: 'background', 1: 'edema', 2: 'enhancing tumor'}  # noqa: E501
    :param dataset_name: The name of the dataset. Can be anything you want
    :param license:
    :param dataset_description:
    :param dataset_reference: website of the dataset, if available
    :param dataset_release:
    :return:
    """  # noqa: E501
    train_identifiers = get_identifiers_from_splitted_files(imagesTr_dir)

    if imagesTs_dir is not None:
        test_identifiers = get_identifiers_from_splitted_files(imagesTs_dir)
    else:
        test_identifiers = []

    json_dict = {
        "name": dataset_name,
        "description": dataset_description,
        "tensorImageSize": "4D",
        "reference": dataset_reference,
        "licence": license,
        "release": dataset_release,
        "modality": {str(i): modalities[i] for i in range(len(modalities))},
        "labels": labels,  # {str(i): labels[i] for i in labels.keys()},
        "numTraining": len(train_identifiers),
        "numTest": len(test_identifiers),
        "training": [{"image": "./imagesTr/%s_0000.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in train_identifiers],
        "test": ["./imagesTs/%s.nii.gz" % i for i in test_identifiers],
    }

    if len(modalities) > 1 or n_tasks > 1:
        training_list = []
        for i in train_identifiers:
            training_case = {}
            for id, modality in enumerate(modalities):
                id_modality = "{0:04d}".format(id)
                training_case["image_{}".format(id)] = "./imagesTr/{}_{}.nii.gz".format(i, id_modality)
            for id, task in enumerate(range(n_tasks)):
                id_task = "{0:04d}".format(id)
                training_case["label_{}".format(id)] = "./labelsTr/{}_{}.nii.gz".format(i, id_task)
            training_list.append(training_case)
        json_dict["training"] = training_list

        testing_list = []
        for i in test_identifiers:
            test_case = {}
            for id, modality in enumerate(modalities):
                id_modality = "{0:04d}".format(id)
                test_case["image_{}".format(id)] = "./imagesTs/{}_{}.nii.gz".format(i, id_modality)

            testing_list.append(test_case)
        json_dict["test"] = testing_list

    if not output_file.endswith("dataset.json"):
        print(
            "WARNING: output file name is not dataset.json! This may be intentional or not. You decide. "  # noqa: E501
            "Proceeding anyways..."
        )
    save_config_json(json_dict, output_file)


def save_config_json(config_dict: Dict[str, str], output_json: str) -> int:
    """

    Parameters
    ----------
    output_json: JSON file path to be saved
    config_dict: dictionary to be saved in JSON format in the RESULTS_FOLDER

    """

    with open(output_json, "w") as fp:
        json.dump(config_dict, fp)
        return 0


def create_nnunet_data_folder_tree(data_folder: str, task_name: str, task_id: str):
    """
    Create nnUNet_raw_data_base folder tree, ready to be populated with the dataset

    :param data_folder: folder path corresponding to the nnUNet_raw_data_base ENV variable
    :param task_id: string used as task_id when creating task folder
    :param task_name: string used as task_name when creating task folder
    """  # noqa E501
    logger.log(DEBUG, ' Creating Dataset tree at "{}"'.format(data_folder))

    Path(data_folder).joinpath("nnUNet_raw_data", "Task" + task_id + "_" + task_name, "imagesTr",).mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(data_folder).joinpath("nnUNet_raw_data", "Task" + task_id + "_" + task_name, "labelsTr",).mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(data_folder).joinpath("nnUNet_raw_data", "Task" + task_id + "_" + task_name, "imagesTs",).mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(data_folder).joinpath("nnUNet_raw_data", "Task" + task_id + "_" + task_name, "labelsTs",).mkdir(
        parents=True,
        exist_ok=True,
    )


def create_data_folder_tree(data_folder: str, task_name: str, task_id: str):
    """
    Create dataset folder tree, ready to be populated with the dataset

    :param data_folder: folder path for the database to be saved
    :param task_id: string used as task_id when creating task folder
    :param task_name: string used as task_name when creating task folder
    """  # noqa E501
    logger.log(DEBUG, ' Creating Dataset tree at "{}"'.format(data_folder))

    Path(data_folder).joinpath("Task" + task_id + "_" + task_name, "imagesTr",).mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(data_folder).joinpath("Task" + task_id + "_" + task_name, "labelsTr",).mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(data_folder).joinpath("Task" + task_id + "_" + task_name, "imagesTs",).mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(data_folder).joinpath("Task" + task_id + "_" + task_name, "labelsTs",).mkdir(
        parents=True,
        exist_ok=True,
    )


def split_dataset(input_data_folder: str, test_split_ratio: int, seed: int) -> Tuple[List[str], List[str]]:
    """

    Parameters
    ----------
    input_data_folder: folder path of the input dataset
    test_split_ratio:  integer value in the range 0-100, specifying the split ratio to be used for the test set
    seed: integer value to be used as Random seed
    Returns
    -------
    train_subjects and test_subjects: lists of strings containing subject IDs for train set and test set respectively

    """  # noqa E501

    subjects = subfolders(input_data_folder, join=False)

    random.seed(seed)
    random.shuffle(subjects)

    split_index = len(subjects) - int(len(subjects) * test_split_ratio / 100)

    train_subjects = subjects[0:split_index]
    test_subjects = subjects[split_index:]

    return train_subjects, test_subjects


def copy_data_to_dataset_folder(
    input_data_folder: str,
    subjects: List[str],
    output_data_folder: str,
    image_subpath: str,
    config_dict: Dict[str, str],
    labels_subpath: str = None,
    num_threads: int = None,
):
    """

    Parameters
    ----------
    num_threads: number of threads to use in multiprocessing ( Default: N_THREADS )
    input_data_folder: folder path of the input dataset
    subjects: string list containing subject IDs for train set
    output_data_folder: folder path where to store images ( and labels )
    modality code following the sequential order.
    image_subpath: relative folder name where to store images in nnUNet folder hierarchy: imagesTr/imagesTs
    are not stored
    labels_subpath: relative folder name where to store labels in nnUNet folder hierarchy ( Default: None ).
    If label_suffix is None, labels are not stored
    config_dict: dictionary with dataset and nnUNet configuration parameters
    """
    label_suffix = config_dict["label_suffix"]
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
            suffix=config_dict["FileExtension"],
        )
        image_suffix_list = config_dict["Modalities"].keys()
        for modality, image_suffix in enumerate(image_suffix_list):
            modality_code = "_{0:04d}".format(modality)
            image_filename = directory + image_suffix

            if image_filename in files:
                updated_image_filename = image_filename.replace(image_suffix, modality_code + config_dict["FileExtension"])
                copied_files.append(
                    pool.starmap_async(
                        copy_image_file,
                        (
                            (
                                str(Path(input_data_folder).joinpath(directory, image_filename)),
                                str(Path(output_data_folder).joinpath(image_subpath, updated_image_filename)),
                            ),
                        ),
                    )
                )
            else:
                logger.warning("{} is not found: skipping {} case".format(image_filename, directory))
        if label_suffix is not None and type(label_suffix) != list:

            label_filename = directory + label_suffix

            if label_filename in files:

                updated_label_filename = label_filename.replace(label_suffix, config_dict["FileExtension"])

                copied_files.append(
                    pool.starmap_async(
                        copy_label_file,
                        (
                            (
                                str(Path(input_data_folder).joinpath(directory, directory + image_suffix)),
                                str(Path(input_data_folder).joinpath(directory, directory + label_suffix)),
                                str(Path(output_data_folder).joinpath(labels_subpath, updated_label_filename)),
                            ),
                        ),
                    )
                )
            else:
                logger.warning("{} is not found: skipping {} case".format(label_filename, directory))

        elif type(label_suffix) == list:  # Multi Label
            for task_id, label_s in enumerate(label_suffix):
                task_code = "_{0:04d}".format(task_id)
                label_filename = directory + label_s

                if label_filename in files:

                    updated_label_filename = label_filename.replace(label_s, task_code + config_dict["FileExtension"])

                    copied_files.append(
                        pool.starmap_async(
                            copy_label_file,
                            (
                                (
                                    str(Path(input_data_folder).joinpath(directory, directory + image_suffix)),
                                    str(Path(input_data_folder).joinpath(directory, directory + label_s)),
                                    str(Path(output_data_folder).joinpath(labels_subpath, updated_label_filename)),
                                ),
                            ),
                        )
                    )
                else:
                    logger.warning("{} is not found: skipping {} case".format(label_filename, directory))

    _ = [i.get() for i in tqdm(copied_files)]


def copy_image_file(input_filepath: str, output_filepath: str):
    """

    Parameters
    ----------
    input_filepath: file path for the file to copy
    output_filepath: file path where to copy the file
    """
    shutil.copy(
        input_filepath,
        output_filepath,
    )


def copy_label_file(input_image: str, input_label: str, output_filepath: str):
    """

    Parameters
    ----------
    input_image: file path for the input image, to be used as reference when copying image information
    input_label: file path for the input label to be copied
    output_filepath: file location where to save the label image
    """
    label_nib = nib.load(input_label)
    image_nib = nib.load(input_image)

    label_nib_out = nib.Nifti1Image(label_nib.get_fdata(), image_nib.affine)
    nib.save(label_nib_out, output_filepath)


def move_file_in_subfolders(folder: str, file_suffix: str, file_extension: str):
    """
    Moves all the file with a specified extension from a main folder into the corresponding subfolders, each named with
    the specific file ID. The file ID is given by the filenames ending with *file_pattern*,excluding it.

    Parameters
    ----------
    folder : str
        Main folder from where to move the files
    file_suffix: str
        Suffix used to identify the file IDs
    file_extension: str
        File extension to identify the files to move.
    """
    files = subfiles(folder, join=False, suffix=file_suffix)
    file_IDs = []
    for file in files:
        if file.endswith(file_suffix):
            Path(folder).joinpath(file[: -len(file_suffix)]).mkdir(parents=True, exist_ok=True)
            file_IDs.append(file[: -len(file_suffix)])

    files = subfiles(folder, join=False, suffix=file_extension)
    for file in files:
        for file_id in file_IDs:
            if file_id in file:
                shutil.move(str(Path(folder).joinpath(file)), str(Path(folder).joinpath(file_id, file)))


def match_subject_IDs_by_suffix_length(data_folder: Union[str, PathLike], prefix_length: int) -> List[List[str]]:
    """
    Given a data folder containing subjects subfolders, return a list of grouped subjects (as list ), where the grouped subjects
    share a common initial pattern ID of length **prefix_length**.

    Parameters
    ----------
    data_folder : Union[str, PathLike]
        Data folder path containing subjects subfolders.
    prefix_length : int
        Length of the filename prefix, used to match different subject IDs.
    Returns
    -------
    List[List[str]]
        List of subjects grouped according to the ID prefix.
    """
    subjects = subfolders(data_folder, join=False)
    matching_subjects = []
    for subject in subjects:
        matching_IDs = [matching_ID for matching_ID in subjects if matching_ID[:prefix_length] == subject[:prefix_length]]
        matching_subjects.append(matching_IDs)
    matching_subjects.sort()

    matching_subjects = [k for k, _ in itertools.groupby(matching_subjects)]
    return matching_subjects


def save_4D_volume_as_3D(filename: Union[str, PathLike], suffix: str):
    """
    Save a 4D volume into separated 3D volumes, appending _idx to the filename.

    Parameters
    ----------
    filename : filename: Union[str, PathLike]
        4D volume filepath.
    suffix: str
        filename extension.
    """
    img = nib.load(filename)

    affine_transform = img.affine

    img_array = img.get_fdata()
    img_array = np.transpose(img_array, (3, 0, 1, 2))

    for idx, img_3D in enumerate(img_array):
        image3D = nib.Nifti1Image(img_3D, affine_transform)
        nib.save(image3D, filename[: -len(suffix)] + "_{}".format(idx) + suffix)
