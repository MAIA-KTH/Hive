import os
from typing import Any, List, Dict, Hashable

import SimpleITK as sitk
import numpy as np
from monai.config import KeysCollection
from monai.transforms import MapTransform

from Hive.monai import ORIENTATION_MAP, ACCEPTED_FILE_EXTENSIONS_FOR_2D_SLICES


class Save2DSlicesd(MapTransform):
    """
    Dictionary-based function to get a 3D volume, slice it along the specified orientations and save the 2D slices
     on disk.
     The 2D slices are save in different folders according to the KEY_NAME and the selected ORIENTATION:
     OUTPUT_FOLDER/ORIENTATION/data/KEY_NAME is the folder path where the 2D slices are saved.
     The 2D slices filenames are composed by the 3D volume filename, followed by a sequential 4-digit number,
      indicating the slice number in the 3D volume along the specified ORIENTATION.
     The 2D slices can be saved in 3 different formats:
      -NPY
      -NPZ
      -PNG, in this case it is possible to rescale the pixel values in the range 0-65535 (16 bit).
     A number of lists (one per ORIENTATION) of dictionaries is saved in the meta_dict with name filenames_ORIENTATION.
     Each dict in the list contains a single key,pair value: key=KEY_NAME and
     value=OUTPUT_FOLDER/ORIENTATION/data/KEY_NAME/2D_SLICE_FILENAME
    """

    def __init__(
            self,
            keys: KeysCollection,
            output_folder: str,
            file_extension: str = ".nii.gz",
            slices_2d_filetype: str = ".npz",
            slicing_axes: List[str] = None,
            rescale_to_png: Dict[Hashable, bool] = None,
            allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be sliced.
            output_folder: folder path where to store 2D slices
            file_extension: file extension of the 3D Volume
            slices_2d_filetype: file extension for the 2D slices. Accepted values are .png .npy or .npz
            slicing_axes: list of string, indicating the axis along which to slice the 3D volume. Accepted string values
            for the axes are 'axial', 'coronal' and 'sagittal'
            rescale_to_png: dict indicating for each key if to rescale the 2D PNGs in the 16-bit range
            allow_missing_keys: don't raise exception if key is missing.

        """
        self.output_folder = output_folder
        self.file_extension = file_extension
        self.slicing_axes = slicing_axes
        self.slices_2D_filetype = slices_2d_filetype
        self.rescale_to_PNG = rescale_to_png
        MapTransform.__init__(self, keys, allow_missing_keys)

        if self.slices_2D_filetype not in ACCEPTED_FILE_EXTENSIONS_FOR_2D_SLICES:
            raise ValueError(
                "File extension should be one of the following: {} , got {}".format(
                    ACCEPTED_FILE_EXTENSIONS_FOR_2D_SLICES, self.slices_2D_filetype
                )
            )

        for slicing_a in self.slicing_axes:
            if slicing_a not in ORIENTATION_MAP:
                raise ValueError(
                    "Slicing axes should be one of the following: {} , got {}".format(ORIENTATION_MAP.keys(), slicing_a)
                )

    def __call__(self, data: Any):

        for key in self.keys:

            for orientation in self.slicing_axes:
                orientation_index = ORIENTATION_MAP[orientation]
                data["{}_meta_dict".format(key)]["filenames_" + orientation] = []
                output_folder = os.path.join(self.output_folder, orientation, "data", str(key))
                os.makedirs(
                    output_folder,
                    exist_ok=True,
                )
                axis_to_flip = [axis for axis, flip in enumerate(data["{}_meta_dict".format(key)]["axis_flip"]) if flip]
                data[key] = np.flip(data[key], axis_to_flip)
                axis_index = data["{}_meta_dict".format(key)]["axis_orientation"].index(orientation_index)
                data[key] = np.swapaxes(data[key], 0, axis_index)
                for index, slice_2d in enumerate(data[key]):

                    output_file = os.path.join(
                        output_folder,
                        os.path.basename(data["image_meta_dict"]["filename_or_obj"][: -len(self.file_extension)])
                        + "_{0:04d}".format(index)
                        + "{}".format(self.slices_2D_filetype),
                    )

                    file_already_exist = os.path.isfile(output_file)

                    if self.slices_2D_filetype == ".png" and not file_already_exist:
                        if self.rescale_to_PNG[key]:
                            slice_2d = (slice_2d - np.min(slice_2d)) * 65535 / (np.max(slice_2d) - np.min(slice_2d))

                        itk_image = sitk.GetImageFromArray(slice_2d)
                        itk_image = sitk.Cast(itk_image, sitk.sitkUInt16)
                        sitk.WriteImage(itk_image, output_file)
                    if self.slices_2D_filetype == ".npz" and not file_already_exist:
                        np.savez_compressed(output_file, slice_2d)
                    if self.slices_2D_filetype == ".npy" and not file_already_exist:
                        np.save(output_file, slice_2d)
                    data["{}_meta_dict".format(key)]["filenames_" + orientation].append({key: output_file})
                data[key] = np.swapaxes(data[key], 0, axis_index)
                data[key] = np.flip(data[key], axis_to_flip)

        return data
