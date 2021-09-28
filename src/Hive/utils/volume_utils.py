from os import PathLike
from pathlib import Path
from typing import List, Dict, Tuple, Any, Union

import nibabel as nib
import numpy as np
from monai.transforms import (
    LoadImaged,
)
from nptyping import NDArray
from scipy.ndimage import binary_erosion
from scipy.ndimage import center_of_mass

from Hive.utils.log_utils import get_logger

logger = get_logger(__name__)


def compare_lung_volumes(filename_dict_list: List[Dict[str, PathLike]]) -> List[int]:
    """
    For a List of Filename dictionaries ( containing ``"image"`` as image filepath and ``"label"`` as label filepath),
    return a list of corresponding lung volumes, expressed in L.

    Parameters
    ----------
    filename_dict_list : List[Dict[str, PathLike]]
        List of dictionaries for image filepaths and label filepaths.

    Returns
    -------
    List[int]
        list of lung volumes, expressed in L.
    """
    lung_volume_list = []
    spacing_volume_list = []
    for filename_dict in filename_dict_list:
        data = LoadImaged(keys=["image", "label"])(filename_dict)
        spacing_volume = np.around(data["image_meta_dict"]["pixdim"][1:4], 3)
        spacing_volume_list.append(spacing_volume)
        spacing_mask = np.around(data["label_meta_dict"]["pixdim"][1:4], 3)
        if np.sum(np.abs(spacing_mask - spacing_volume)) > 0:
            logger.info(
                "{}: voxel spacing does not match for image and label. {} and {}".format(
                    filename_dict["image"], spacing_volume, spacing_mask
                )
            )
        voxel_volume = np.sum(data["label"] > 0)
        volume_in_L = (voxel_volume * spacing_volume[0] * spacing_volume[1] * spacing_volume[2]) / 1000000
        lung_volume_list.append(volume_in_L)
    spacing = np.array(spacing_volume_list)
    if not (
        np.all(spacing[:, 0] == spacing[0, 0])
        and np.all(spacing[:, 1] == spacing[0, 1])
        and np.all(spacing[:, 2] == spacing[0, 2])
    ):
        logger.info("{}: voxel spacing does not match. {}".format(filename_dict_list[0]["image"], spacing_volume_list))
    return lung_volume_list


def decompose_affine_transform(
    affine_transform: NDArray[(4, 4), float]
) -> Tuple[NDArray[(3, 3), float], NDArray[(3,), float], NDArray[(3,), float]]:
    """
    Decompose a 4x4 affine transform into the rotation, scale and offset component.

    Parameters
    ----------
    affine_transform : NDArray[(4, 4), float]
        4x4 affine transformation to decompose.

    Returns
    -------
    Tuple[
    NDArray[(3, 3), float], NDArray[(3,), float], NDArray[(3,), float]]
        Rotation matrix, scaling and offset components.
    """
    offset = affine_transform[:3, -1]
    scale = np.sqrt(np.sum(np.square(affine_transform[:3, :3]), axis=0))
    rotation = affine_transform[:3, :3] / scale

    return rotation, scale, offset


def apply_affine_transform_to_vector_field(
    vector_field: NDArray[(3, Any, Any, Any), float], affine_transform: NDArray[(4, 4), float]
) -> NDArray[(3, Any, Any, Any), float]:
    """
    Apply affine transform to a 3-channel 3D volume, representing x-y-z coordinates in the channel dimension.
    The vector field should be in the form CHWD.

    Parameters
    ----------
    vector_field : NDArray[(3,Any,Any,Any), float]
        4D Array, in the form CHWD, with C=3.
    affine_transform : NDArray[(4,4), float]
        4x4 Affine transform
    Returns
    -------
    NDArray[(3,Any,Any,Any), float]

        Transformed vector field.
    """
    if np.sum(affine_transform - np.eye(4)) == 0:
        return vector_field

    vector_field_4D = np.append(
        vector_field, np.zeros((1, vector_field.shape[1], vector_field.shape[2], vector_field.shape[3])), axis=0
    )

    transformed_vector_field = np.tensordot(np.transpose(vector_field_4D, axes=[1, 2, 3, 0]), affine_transform, axes=1)

    transformed_vector_field = np.transpose(transformed_vector_field, axes=[3, 0, 1, 2])

    return transformed_vector_field[:3, :]


def compute_label_volumes(
    label_array: NDArray[(Any, Any, Any), int], image_meta_dict: Dict[str, Any], labels: List[int]
) -> List[float]:
    """
    Computes and returns Volume size (in L), for each label listed and found in the array.

    Parameters
    ----------
    label_array : NDArray[(Any,Any,Any), int]
        3D label array.
    image_meta_dict : Dict[str, Any]
        Metadata dictionary of the label image.
    labels : List[int]
        List of labels to consider.

    Returns
    -------
    List[float]
        Volume size (expressed in L), for each listed label.
    """
    label_volumes = []
    spacing = image_meta_dict["pixdim"][1:4]
    for label in labels:
        label_volume = (np.sum(label_array == label) * spacing[0] * spacing[1] * spacing[2]) / 1000000
        label_volumes.append(label_volume)
    return label_volumes


def compute_label_avg_intensity(
    image_array: NDArray[(Any, Any, Any), int], label_array: NDArray[(Any, Any, Any), int], labels: List[int]
) -> List[float]:
    """
    Computes and returns voxel average intensity for each label listed and found in the array.

    Parameters
    ----------
    image_array : NDArray[(Any,Any,Any), int]
        3D Array containing voxel intensities.
    label_array : NDArray[(Any,Any,Any), int]
        3D Array containing voxel labels.
    labels : List[int]
        List of labels to consider.

    Returns
    -------
    List[float]
        Average voxel intensities for each listed label.
    """
    label_avg_intensity = []
    for label in labels:
        masked_array = np.ma.array(image_array, mask=label_array != label)
        label_avg_intensity.append(masked_array.mean())
    return label_avg_intensity


def compute_center_of_mass(
    label_array: NDArray[(Any, Any, Any), int], image_meta_dict: Dict[str, Any], labels: List[int]
) -> List[NDArray[(3,), float]]:
    """
    Computes and returns labels center of mass.

    Parameters
    ----------
    label_array : NDArray[(Any,Any,Any), int]
        3D array.
    image_meta_dict: Dict[str, Any]
        Metadata dictionary of the label image.
    labels : List[int]
        List of labels to consider.

    Returns
    -------
    List[NDArray[(3,), float]]
        List of 3D Numpy array. containing label center of mass.
    """
    affine = image_meta_dict["affine"]
    affine[0, :] = -affine[0, :]
    affine[1, :] = -affine[1, :]
    label_cm = center_of_mass(label_array > 0, label_array, labels)
    label_cm = [np.dot(affine, np.append(cm, 0))[:3] for cm in label_cm]
    return label_cm


def compute_subject_summary(filename_dict: Dict[str, Union[str, PathLike]], label_dict: Dict[str, str]) -> Dict[str, str]:
    """
    Computes and returns volume and label summary, for the given subject. The subject is specified in the **filename_dict**,
    where the corresponding image and label filepaths are indicated. The labels of interest are specified in **label_dict**.
    Parameters
    ----------
    filename_dict : Dict[str, Union[str, PathLike]]
        Dictionary containing "image" and "label" keys, indicating the corresponding filepaths.
    label_dict : Dict[str, str]
        Label dictionary, containing the labels as keys.

    Returns
    -------
    Dict[str, str]
        Dictionary containing label and volume information for the specified subject.
    """
    data = LoadImaged(keys=["image", "label"])(filename_dict)
    subject_summary = {}
    labels = [int(label) for label in label_dict]
    labels = labels[1:]
    subject_summary["ID"] = Path(filename_dict["image"]).parent.name
    subject_summary["Resolution"] = data["image_meta_dict"]["pixdim"][1:4]
    subject_summary["Dimension"] = data["image_meta_dict"]["dim"][1:4]
    label_CM = compute_center_of_mass(data["label"], data["image_meta_dict"], labels)
    label_avg_intensity = compute_label_avg_intensity(data["image"], data["label"], labels)
    label_volume = compute_label_volumes(data["label"], data["image_meta_dict"], labels)
    for index, label in enumerate(labels):
        subject_summary["CM_{}".format(label_dict[str(label)])] = label_CM[index]
        subject_summary["Average Intensity_{}".format(label_dict[str(label)])] = label_avg_intensity[index]
        subject_summary["Lobe_Volumes_{}".format(label_dict[str(label)])] = label_volume[index]

    return subject_summary


def erode_mask(filename_dict: Dict[str, Union[str, PathLike]], iterations: int, output_filename: Union[str, PathLike]):
    """
    Given a label mask, performs binary erosion and save the output.

    Parameters
    ----------
    filename_dict : Dict[str, Union[str, PathLike]]
        Dictionary containing"label" key, indicating the corresponding filepath.
    iterations : int
        Number of the itarations for the binary erosion operation.
    output_filename : Union[str, PathLike]
        Filepath where to save the eroded mask.
    """
    data = LoadImaged(keys=["label"])(filename_dict)
    eroded_label = binary_erosion(data["label"], iterations=iterations)
    eroded_nib_label = nib.Nifti1Image(eroded_label.astype(np.int32), data["label_meta_dict"]["affine"])
    nib.save(eroded_nib_label, output_filename)
