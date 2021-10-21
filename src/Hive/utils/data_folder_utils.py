import os
from distutils.dir_util import copy_tree
from os import PathLike
from pathlib import Path
from typing import Union, List

from Hive.utils.log_utils import (
    get_logger,
    WARN,
    DEBUG,
    INFO,
)

logger = get_logger(__name__)


def order_data_in_single_folder(root_path: Union[str, PathLike], output_path: Union[str, PathLike]):
    """
    Moves all the sub-files, found iteratively from the root directory, to the output folder.
    Recursively removes all the empty sub-directories.

    Parameters
    ----------
    root_path   :  Union[str, PathLike]
        Root folder.
    output_path : Union[str, PathLike]
        Output folder.
    """
    logger.log(DEBUG, "Creating folder at '{}'".format(output_path))
    for file_path in Path(root_path).glob("*/*"):
        logger.log(DEBUG, "Moving '{}' file to '{}'".format(file_path, Path(output_path).joinpath(Path(file_path).name)))
        Path(file_path).rename(Path(output_path).joinpath(Path(file_path).name))
    remove_empty_folder_recursive(root_path)


def remove_empty_folder_recursive(folder_path: Union[str, PathLike]):
    """
    Recursively removes all the empty sub-directories of the root folder.

    Parameters
    ----------
    folder_path : Union[str, PathLike]
        Root folder path.
    """
    for subfolder_path in Path(folder_path).glob("*"):
        if Path(subfolder_path).is_dir():
            try:
                os.rmdir(subfolder_path)
            except FileNotFoundError as e:
                logger.log(WARN, e)
            except OSError as e:
                logger.log(WARN, e)
                remove_empty_folder_recursive(subfolder_path)
                os.rmdir(subfolder_path)


def order_data_folder_by_patient(folder_path: Union[str, PathLike], file_pattern: str):
    """
    Order all the files in the root folder into corresponding sub-directories, according to the specified
    file pattern.

    Parameters
    ----------
    folder_path : Union[str, PathLike]
        Root folder path.
    file_pattern    : str
        File pattern to group the files and create the corresponding sub-directories.
    """
    patient_id_list = []
    for file_path in Path(folder_path).glob("*"):
        if Path(file_path).is_file() and str(file_path).endswith(file_pattern):
            patient_id_list.append(str(file_path.name)[: -len(file_pattern)])

    logger.log(INFO, "Patient folders in database: {}".format(len(patient_id_list)))

    for patient_id in patient_id_list:
        logger.log(DEBUG, "Creating folder at '{}'".format(Path(folder_path).joinpath(patient_id)))
        Path(folder_path).joinpath(patient_id).mkdir(exist_ok=True, parents=True)

    for file_path in Path(folder_path).glob("*"):
        if Path(file_path).is_file():
            for patient_id in patient_id_list:

                if file_path.name.startswith(patient_id):
                    logger.log(
                        DEBUG,
                        "Moving '{}' file to '{}'".format(
                            file_path, Path(folder_path).joinpath(patient_id, Path(file_path).name)
                        ),
                    )
                    Path(file_path).rename(Path(folder_path).joinpath(patient_id, Path(file_path).name))


def copy_subject_folder_to_data_folder(
    input_data_folder: Union[str, PathLike], subjects: List[str], data_folder: Union[str, PathLike]
):
    """
    Copy all the specified subject sub-folders to a new data folder.

    Parameters
    ----------
    input_data_folder : Union[str, PathLike]
        Input data folder.
    subjects    : List[str]
        Subjects to copy.
    data_folder : Union[str, PathLike]
        Destination data folder.
    """
    Path(data_folder).mkdir(parents=True, exist_ok=True)
    for subject in subjects:
        if Path(input_data_folder).joinpath(subject).is_dir():
            logger.log(DEBUG, "Copying Subject {}".format(subject))
            copy_tree(str(Path(input_data_folder).joinpath(subject)), str(Path(data_folder).joinpath(subject)))
