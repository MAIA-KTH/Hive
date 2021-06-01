import json
import os
import random
import shutil
from typing import Dict, Tuple, List

import SimpleITK as sitk

from .log_utils import get_logger

logger = get_logger(__name__)


def save_config_json(config_dict: Dict[str, str]) -> int:
    """

    Parameters
    ----------
    config_dict: dictionary to be saved in JSON format in the RESULTS_FOLDER


    """
    output_json_basename = (
            config_dict["DatasetName"]
            + "_"
            + config_dict["TRAINING_CONFIGURATION"]
            + "_"
            + config_dict["Task_ID"]
            + "_"
            + config_dict["Task_Name"]
            + ".json"
    )
    try:
        output_json = os.path.join(os.environ["RESULTS_FOLDER"], output_json_basename)

        config_dict["results_folder"] = os.environ["RESULTS_FOLDER"]
        try:
            config_dict["preprocessing_folder"] = os.environ["nnUNet_preprocessed"]
        except KeyError:
            logger.warning(
                "nnUNet_preprocessed is not set as environment variable, not saved in {}".format(  # noqa E501
                    output_json_basename
                )
            )

        with open(output_json, "w") as fp:
            json.dump(config_dict, fp)
        return 0
    except KeyError:
        logger.warning(
            "RESULTS_FOLDER is not set as environment variable, {} is not saved".format(
                output_json_basename
            )
        )
        return 1


def create_nnunet_data_folder_tree(data_folder: str, task_name: str, task_id: str):
    """
    Create nnUNet_raw_data_base folder tree, ready to be populated with the dataset

    :param data_folder: folder path corresponding to the nnUNet_raw_data_base ENV variable
    :param task_id: string used as task_id when creating task folder
    :param task_name: string used as task_name when creating task folder
    """  # noqa E501
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


def split_dataset(
        input_data_folder: str, test_split_ratio: int
) -> Tuple[List[str], List[str]]:
    """

    Parameters
    ----------
    input_data_folder: folder path of the input dataset
    test_split_ratio:  integer value in the range 0-100, specifying the split ratio to be used for the test set

    Returns
    -------
    train_subjects and test_subjects: lists of strings containing subject IDs for train set and test set respectively

    """  # noqa E501
    subject = [dirs for _, dirs, _ in os.walk(input_data_folder)]
    subjects = subject[0]  # TODO: Refactor subdirectory listing

    random.seed(6)
    random.shuffle(subjects)

    split_index = len(subjects) - int(len(subjects) * test_split_ratio / 100)

    train_subjects = subjects[0:split_index]
    test_subjects = subjects[split_index:]

    return train_subjects, test_subjects


def copy_images_to_nnunet_train_data_folder(
        input_data_folder: str,
        train_subjects: List[str],
        nnunet_data_folder: str,
        image_suffix: str,
        label_suffix: str,
        config_dict: Dict[str, str],
        modality: int = 0,
):
    """

    Parameters
    ----------
    input_data_folder: folder path of the input dataset
    train_subjects: string list containing subject IDs for train set
    nnunet_data_folder: folder path for nnUNet base dataset, corresponding to nnUNet_raw_data_base env variable
    image_suffix: file suffix to be used to correctly detect the file to store in imagesTr
    label_suffix: file suffix to be used to correctly detect the file to store in labelsTr
    config_dict: dictionary with dataset and nnUNet configuration parameters
    modality: integer value indexing the modality in config_dict['modalities'] to be considered ( Default: 0 in single modality ) # noqa: E501
    """

    modality_code = "{0:04d}".format(modality)
    for directory in train_subjects:
        for _, _, files in os.walk(os.path.join(input_data_folder, directory)):
            for (
                    file
            ) in files:  # TODO : debug log to check if image+label mask are found

                if file == (directory + image_suffix):
                    nnunet_image_filename = file.replace(
                        image_suffix, "_" + modality_code + config_dict["FileExtension"]
                    )
                    shutil.copy(
                        os.path.join(input_data_folder, directory, file),
                        os.path.join(
                            nnunet_data_folder, "imagesTr", nnunet_image_filename
                        ),
                    )
                if file == (directory + label_suffix):
                    nnunet_label_filename = file.replace(
                        label_suffix, config_dict["FileExtension"]
                    )
                    image_1 = sitk.ReadImage(
                        os.path.join(
                            input_data_folder, directory, directory + label_suffix
                        )
                    )
                    image_2 = sitk.ReadImage(
                        os.path.join(
                            input_data_folder, directory, directory + image_suffix
                        )
                    )
                    image_1.CopyInformation(image_2)
                    sitk.WriteImage(
                        image_1,
                        os.path.join(
                            nnunet_data_folder, "labelsTr", nnunet_label_filename
                        ),
                    )


def copy_images_to_nnunet_test_data_folder(
        input_data_folder: str,
        test_subjects: List[str],
        nnunet_data_folder: str,
        image_suffix: str,
        config_dict: Dict[str, str],
        modality: int = 0,
):
    """

    Parameters
    ----------
    input_data_folder: folder path of the input dataset
    test_subjects: string list containing subject IDs for test set
    nnunet_data_folder: folder path for nnUNet base dataset, corresponding to nnUNet_raw_data_base env variable
    image_suffix: file suffix to be used to correctly detect the file to store in imagesTs
    config_dict: dictionary with dataset and nnUNet configuration parameters
    modality: integer value indexing the modality in config_dict['modalities'] to be considered ( Default: 0 in single modality )
    """  # noqa E501
    modality_code = "{0:04d}".format(modality)

    for directory in test_subjects:
        for _, _, files in os.walk(os.path.join(input_data_folder, directory)):
            for file in files:

                if file == (directory + image_suffix):
                    nnunet_image_filename = file.replace(
                        image_suffix, "_" + modality_code + config_dict["FileExtension"]
                    )
                    shutil.copy(
                        os.path.join(input_data_folder, directory, file),
                        os.path.join(
                            nnunet_data_folder, "imagesTs", nnunet_image_filename
                        ),
                    )
