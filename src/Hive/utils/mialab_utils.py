import subprocess
from os import PathLike
from pathlib import Path
from typing import Union

from Hive.utils.log_utils import DEBUG, get_logger

logger = get_logger(__name__)


def run_mialab_fuzzy_segmentation_command(
    mialab_folder: Union[str, PathLike],
    data_folder: Union[str, PathLike],
    subject: str,
    image_suffix: str,
    output_suffix: str,
    mask_suffix: str,
):
    """
    Run MiaLab Fuzzy Connectedness Segmentation operation on given subject, saving the Connectivity vector field as output.

    Parameters
    ----------
    mialab_folder : Union[str, PathLike]
        Folder to locate MiaLab executable.
    data_folder : Union[str, PathLike]
        Dataset folder.
    subject : str
        Subject ID.
    image_suffix : str
        String suffix to detect input image filename.
    output_suffix : str
        String suffix to represent output image filename.
    mask_suffix : str
        String suffix to detect mask image filename.
    """
    command = str(Path(mialab_folder).joinpath("MiaLab")) + " -op FuzzyCon"
    command += " -InputImage {}".format(Path(data_folder).joinpath(subject, subject + image_suffix))
    command += " -MaskImage {}".format(Path(data_folder).joinpath(subject, subject + mask_suffix))
    command += " -OutputImage {}".format(Path(data_folder).joinpath(subject, subject + output_suffix))
    logger.log(DEBUG, "Running {}".format(command))
    subprocess.run(command)


def run_mialab_lung_lobe_annotation_command(
    mialab_folder: Union[str, PathLike],
    data_folder: Union[str, PathLike],
    subject: str,
    file_extension: str,
):
    """
    Run MiaLab Lung Lobe TPS tool on given subject, saving the lung and lobe masks as output.

    Parameters
    ----------
    file_extension : str
        File extension to fully identify the input file (e.g. ``".nii.gz"``).
    subject : str
        Subject ID
    mialab_folder : Union[str, PathLike]
        Folder to locate MiaLab executable.
    data_folder : Union[str, PathLike]
        Dataset folder.

    """
    command = str(Path(mialab_folder).joinpath("MiaLab")) + " -wl LungLobeTPSSegmentation"
    command += " -InputImage {}".format(Path(data_folder).joinpath(subject, subject + file_extension))

    logger.log(DEBUG, "Running {}".format(command))
    subprocess.run(command)
