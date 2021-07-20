import json
import os
import random
import shutil
from multiprocessing import Pool
from typing import Dict, Tuple, List

import SimpleITK as sitk
import numpy as np

from .log_utils import get_logger, DEBUG

logger = get_logger(__name__)


def subfiles(folder, join=True, prefix=None, suffix=None, sort=True):
    if join:
        l = os.path.join  # noqa: E741
    else:
        l = lambda x, y: y  # noqa: E741, E731
    res = [
        l(folder, i)
        for i in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, i))
        and (prefix is None or i.startswith(prefix))
        and (suffix is None or i.endswith(suffix))
    ]
    if sort:
        res.sort()
    return res


def subfolders(folder, join=True, sort=True):
    if join:
        l = os.path.join  # noqa: E741
    else:
        l = lambda x, y: y  # noqa: E741, E731
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isdir(os.path.join(folder, i))]
    if sort:
        res.sort()
    return res


def get_identifiers_from_splitted_files(folder: str):
    uniques = np.unique([i for i in subfiles(folder, suffix=".nii.gz", join=False)])
    return uniques


def generate_dataset_json(
    output_file: str,
    imagesTr_dir: str,
    imagesTs_dir: str,
    modalities: Tuple,
    labels: dict,
    dataset_name: str,
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

    json_dict = {}
    json_dict["name"] = dataset_name
    json_dict["description"] = dataset_description
    json_dict["tensorImageSize"] = "4D"
    json_dict["reference"] = dataset_reference
    json_dict["licence"] = license
    json_dict["release"] = dataset_release
    json_dict["modality"] = {str(i): modalities[i] for i in range(len(modalities))}
    json_dict["labels"] = {str(i): labels[i] for i in labels.keys()}

    json_dict["numTraining"] = len(train_identifiers)
    json_dict["numTest"] = len(test_identifiers)
    json_dict["training"] = [{"image": "./imagesTr/%s" % i, "label": "./labelsTr/%s" % i} for i in train_identifiers]
    json_dict["test"] = ["./imagesTs/%s" % i for i in test_identifiers]

    if not output_file.endswith("dataset.json"):
        print(
            "WARNING: output file name is not dataset.json! This may be intentional or not. You decide. "  # noqa: E501
            "Proceeding anyways..."
        )
    save_config_json(json_dict, os.path.join(output_file))


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
    os.makedirs(
        os.path.join(
            data_folder,
            "nnUNet_raw_data",
            "Task" + task_id + "_" + task_name,
            "imagesTr",
        ),
        exist_ok=True,
    )
    os.makedirs(
        os.path.join(
            data_folder,
            "nnUNet_raw_data",
            "Task" + task_id + "_" + task_name,
            "labelsTr",
        ),
        exist_ok=True,
    )
    os.makedirs(
        os.path.join(
            data_folder,
            "nnUNet_raw_data",
            "Task" + task_id + "_" + task_name,
            "imagesTs",
        ),
        exist_ok=True,
    )

    os.makedirs(
        os.path.join(
            data_folder,
            "nnUNet_raw_data",
            "Task" + task_id + "_" + task_name,
            "labelsTs",
        ),
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
    os.makedirs(
        os.path.join(
            data_folder,
            "Task" + task_id + "_" + task_name,
            "imagesTr",
        ),
        exist_ok=True,
    )
    os.makedirs(
        os.path.join(
            data_folder,
            "Task" + task_id + "_" + task_name,
            "labelsTr",
        ),
        exist_ok=True,
    )
    os.makedirs(
        os.path.join(
            data_folder,
            "Task" + task_id + "_" + task_name,
            "imagesTs",
        ),
        exist_ok=True,
    )

    os.makedirs(
        os.path.join(
            data_folder,
            "Task" + task_id + "_" + task_name,
            "labelsTs",
        ),
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
    train_subjects: List[str],
    output_data_folder: str,
    image_suffix: str,
    image_subpath: str,
    config_dict: Dict[str, str],
    label_suffix: str = None,
    labels_subpath: str = None,
    modality: int = None,
    num_threads: int = 5,
):
    """

    Parameters
    ----------
    num_threads: number of threads to use in multiprocessing ( Default: 5 )
    input_data_folder: folder path of the input dataset
    train_subjects: string list containing subject IDs for train set
    output_data_folder: folder path where to store images ( and labels )
    image_suffix: file suffix to be used to correctly detect the file to store in imagesTr/imagesTs
    image_subpath: relative folder name where to store images in nnUNet folder hierarchy: imagesTr/imagesTs
    label_suffix: file suffix to be used to correctly detect the file to store in labelsTr. If None, label images
    are not stored
    labels_subpath: relative folder name where to store labels in nnUNet folder hierarchy ( Default: None ). If label_suffix is None,
    labels are not stored
    config_dict: dictionary with dataset and nnUNet configuration parameters
    modality: integer value indexing the modality in config_dict['modalities'] to be considered ( Default: None ). # noqa: E501
              If None, no modality code is appended
    """

    if modality is not None:
        modality_code = "_{0:04d}".format(modality)
    else:
        modality_code = ""

    pool = Pool(num_threads)
    copied_files = []
    for directory in train_subjects:

        files = subfiles(
            os.path.join(input_data_folder, directory),
            join=False,
            suffix=config_dict["FileExtension"],
        )

        image_filename = directory + image_suffix

        if label_suffix is not None:

            label_filename = directory + label_suffix

            if image_filename in files and label_filename in files:
                updated_image_filename = image_filename.replace(image_suffix, modality_code + config_dict["FileExtension"])
                updated_label_filename = label_filename.replace(label_suffix, config_dict["FileExtension"])
                copied_files.append(
                    pool.starmap_async(
                        copy_image_file,
                        (
                            (
                                os.path.join(input_data_folder, directory, image_filename),
                                os.path.join(output_data_folder, image_subpath, updated_image_filename),
                            ),
                        ),
                    )
                )
                copied_files.append(
                    pool.starmap_async(
                        copy_label_file,
                        (
                            (
                                os.path.join(input_data_folder, directory, directory + image_suffix),
                                os.path.join(input_data_folder, directory, directory + label_suffix),
                                os.path.join(output_data_folder, labels_subpath, updated_label_filename),
                            ),
                        ),
                    )
                )

            else:
                logger.warning("{} or {} are not stored: skipping {} case".format(image_filename, label_filename, directory))
        else:
            updated_image_filename = image_filename.replace(image_suffix, modality_code + config_dict["FileExtension"])
            copied_files.append(
                pool.starmap_async(
                    copy_image_file,
                    (
                        (
                            os.path.join(input_data_folder, directory, image_filename),
                            os.path.join(output_data_folder, image_subpath, updated_image_filename),
                        ),
                    ),
                )
            )


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
    label_itk = sitk.ReadImage(input_label)
    image_itk = sitk.ReadImage(input_image)
    label_itk.CopyInformation(image_itk)
    sitk.WriteImage(label_itk, output_filepath)
