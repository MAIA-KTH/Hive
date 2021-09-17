from os import PathLike
from typing import List, Dict

import numpy as np
from monai.transforms import (
    LoadImaged,
)


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
    for filename_dict in filename_dict_list:
        data = LoadImaged(keys=["image", "label"])(filename_dict)
        spacing = data["image_meta_dict"]["pixdim"][1:4]
        voxel_volume = np.sum(data["label"] > 0)
        volume_in_L = (voxel_volume * spacing[0] * spacing[1] * spacing[2]) / 1000000
        lung_volume_list.append(volume_in_L)

    return lung_volume_list
