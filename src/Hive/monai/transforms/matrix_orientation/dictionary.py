import copy
from typing import Any

import numpy as np
from Hive.monai import ORIENTATION_MAP
from Hive.monai.transforms.utils import get_axis_order_to_RAI
from monai.config import KeysCollection
from monai.transforms import MapTransform


class OrientToRAId(MapTransform):
    """
    Dictionary-based transform, used to include information about the axis orientation in the meta_dict.
    If slicing_axes is given as parameter, the 3D volume axis are permuted in order to represent the corresponding
    orientation along the first dimension.
    Raises:
        ValueError: When q_form or s_form are not used in the NIFTI metadata to describe the rotation matrix
    """

    def __init__(
        self,
        keys: KeysCollection,
        slicing_axes: str = None,
        inverse_function: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
            slicing_axes: optional string, indicating the orientation to be displayed on the first dimension.
            Accepted string values for the axes are 'axial', 'coronal' and 'sagittal'. If None (Default), no permutation
            is performed.
            inverse_function: reverse order of axis flip and transpose.
            allow_missing_keys: don't raise exception if key is missing.

        """

        self.slicing_axes = slicing_axes
        self.inverse_function = inverse_function
        MapTransform.__init__(self, keys, allow_missing_keys)

        if self.slicing_axes not in ORIENTATION_MAP and self.slicing_axes is not None:
            raise ValueError(
                "Slicing axes should be one of the following: {} , got {}".format(ORIENTATION_MAP.keys(), self.slicing_axes)
            )

    def __call__(self, data: Any):

        for key in self.keys:
            affine = copy.deepcopy(data["{}_meta_dict".format(key)]["original_affine"])
            orientation_matrix = np.eye(3)
            orientation_matrix[0] = affine[0][:-1] / -data["{}_meta_dict".format(key)]["pixdim"][1]
            orientation_matrix[1] = affine[1][:-1] / -data["{}_meta_dict".format(key)]["pixdim"][2]
            orientation_matrix[2] = affine[2][:-1] / data["{}_meta_dict".format(key)]["pixdim"][3]
            axis_orientation, flip_axes = get_axis_order_to_RAI(orientation_matrix)
            data["{}_meta_dict".format(key)]["axis_orientation"] = axis_orientation
            data["{}_meta_dict".format(key)]["axis_flip"] = flip_axes
            data["{}_meta_dict".format(key)]["rotation_affine"] = orientation_matrix

            if self.slicing_axes is not None:
                orientation_index = ORIENTATION_MAP[self.slicing_axes]
                axis_index = data["{}_meta_dict".format(key)]["axis_orientation"].index(orientation_index)
                data["{}_meta_dict".format(key)]["axis_orientation_swapped"] = axis_orientation.copy()
                data["{}_meta_dict".format(key)]["axis_orientation_swapped"][0] = axis_orientation[axis_index]
                data["{}_meta_dict".format(key)]["axis_orientation_swapped"][axis_index] = axis_orientation[0]
                axis_to_flip = [axis for axis, flip in enumerate(data["{}_meta_dict".format(key)]["axis_flip"]) if flip]
                if not self.inverse_function:
                    data[key] = np.flip(data[key], axis_to_flip)
                data[key] = np.swapaxes(data[key], 0, axis_index)
                if self.inverse_function:
                    data[key] = np.flip(data[key], axis_to_flip)
                data["{}_meta_dict".format(key)]["spatial_shape"] = data[key].shape
                affine = data["{}_meta_dict".format(key)]["original_affine"]
                affine[0][:-1] = affine[0][:-1] / data["{}_meta_dict".format(key)]["pixdim"][1]
                affine[1][:-1] = affine[1][:-1] / data["{}_meta_dict".format(key)]["pixdim"][2]
                affine[2][:-1] = affine[2][:-1] / data["{}_meta_dict".format(key)]["pixdim"][3]
                data["{}_meta_dict".format(key)]["pixdim"][1], data["{}_meta_dict".format(key)]["pixdim"][axis_index + 1] = (
                    data["{}_meta_dict".format(key)]["pixdim"][axis_index + 1],
                    data["{}_meta_dict".format(key)]["pixdim"][1],
                )
                affine[0][:-1] = affine[0][:-1] * data["{}_meta_dict".format(key)]["pixdim"][1]
                affine[1][:-1] = affine[1][:-1] * data["{}_meta_dict".format(key)]["pixdim"][2]
                affine[2][:-1] = affine[2][:-1] * data["{}_meta_dict".format(key)]["pixdim"][3]
                data["{}_meta_dict".format(key)]["original_affine"] = affine
                data["{}_meta_dict".format(key)]["affine"] = affine
        return data


class TransposeAxesd(MapTransform):
    """
    Dictionary-based transform, used to reverse the order of the axis, permuting to (2,1,0). Used when saving a numpy array
    as image with SimpleITK
    """

    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
            Accepted string values for the axes are 'axial', 'coronal' and 'sagittal'. If None (Default), no permutation
            is performed.
            allow_missing_keys: don't raise exception if key is missing.

        """
        MapTransform.__init__(self, keys, allow_missing_keys)

    def __call__(self, data: Any):

        for key in self.keys:
            data[key] = np.transpose(data[key], (2, 1, 0))

            affine = data["{}_meta_dict".format(key)]["original_affine"]
            affine[0][:-1] = affine[0][:-1] / data["{}_meta_dict".format(key)]["pixdim"][1]
            affine[1][:-1] = affine[1][:-1] / data["{}_meta_dict".format(key)]["pixdim"][2]
            affine[2][:-1] = affine[2][:-1] / data["{}_meta_dict".format(key)]["pixdim"][3]
            data["{}_meta_dict".format(key)]["pixdim"][1], data["{}_meta_dict".format(key)]["pixdim"][3] = (
                data["{}_meta_dict".format(key)]["pixdim"][3],
                data["{}_meta_dict".format(key)]["pixdim"][1],
            )
            affine[0][:-1] = affine[0][:-1] * data["{}_meta_dict".format(key)]["pixdim"][1]
            affine[1][:-1] = affine[1][:-1] * data["{}_meta_dict".format(key)]["pixdim"][2]
            affine[2][:-1] = affine[2][:-1] * data["{}_meta_dict".format(key)]["pixdim"][3]
            data["{}_meta_dict".format(key)]["original_affine"] = affine
            data["{}_meta_dict".format(key)]["affine"] = affine
            data["{}_meta_dict".format(key)]["spatial_shape"] = data[key].shape
        return data
