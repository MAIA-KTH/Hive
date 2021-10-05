from typing import Any

import numpy as np
from monai.config import KeysCollection
from monai.transforms import MapTransform
from scipy.spatial.transform import Rotation as R

from Hive.monai import ORIENTATION_MAP
from Hive.monai.transforms.utils import get_quatern_a, get_axis_order_to_RAI


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
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
            slicing_axes: optional string, indicating the orientation to be displayed on the first dimension.
            Accepted string values for the axes are 'axial', 'coronal' and 'sagittal'. If None (Default), no permutation
            is performed.
            allow_missing_keys: don't raise exception if key is missing.

        """

        self.slicing_axes = slicing_axes

        MapTransform.__init__(self, keys, allow_missing_keys)

        if self.slicing_axes not in ORIENTATION_MAP and self.slicing_axes is not None:
            raise ValueError(
                "Slicing axes should be one of the following: {} , got {}".format(ORIENTATION_MAP.keys(), self.slicing_axes)
            )

    def __call__(self, data: Any):

        for key in self.keys:
            if data["{}_meta_dict".format(key)]["qform_code"] > 0:
                quaterns = [
                    data["{}_meta_dict".format(key)]["quatern_b"],
                    data["{}_meta_dict".format(key)]["quatern_c"],
                    data["{}_meta_dict".format(key)]["quatern_d"],
                ]
                quaterns.insert(0, get_quatern_a(*quaterns))
                r = R.from_quat(quaterns)
                orientation_matrix = r.as_matrix()
            elif data["{}_meta_dict".format(key)]["sform_code"] > 0:

                orientation_matrix = np.array(
                    [
                        data["{}_meta_dict".format(key)]["srow_x"][:-1],
                        data["{}_meta_dict".format(key)]["srow_y"][:-1],
                        data["{}_meta_dict".format(key)]["srow_z"][:-1],
                    ]
                )
            else:
                raise ValueError("Q Form or S Form are not used to describe the rotation matrix in the NIFTI volume")
            axis_orientation, flip_axes = get_axis_order_to_RAI(orientation_matrix)
            data["{}_meta_dict".format(key)]["axis_orientation"] = axis_orientation
            data["{}_meta_dict".format(key)]["axis_flip"] = flip_axes
            data["{}_meta_dict".format(key)]["rotation_affine"] = orientation_matrix

            if self.slicing_axes is not None:
                orientation_index = ORIENTATION_MAP[self.slicing_axes]
                axis_index = data["{}_meta_dict".format(key)]["axis_orientation"].index(orientation_index)
                axis_to_flip = [axis for axis, flip in enumerate(data["{}_meta_dict".format(key)]["axis_flip"]) if flip]
                data[key] = np.flip(data[key], axis_to_flip)
                data[key] = np.swapaxes(data[key], 0, axis_index)

        return data
