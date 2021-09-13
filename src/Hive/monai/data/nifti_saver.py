from typing import Union, Optional, Dict

import numpy as np
import torch
from monai.data import NiftiSaver
from monai.data.nifti_writer import write_nifti


class HiveNiftiSaver(NiftiSaver):
    def __init__(
            self,
            output_dir: str = "./",
            output_postfix: str = "seg",
    ):
        super().__init__(output_dir, output_postfix)

    def set_output_dir(self, output_dir):
        self.output_dir = output_dir

    def set_output_postfix(self, output_postfix):
        self.output_postfix = output_postfix

    def save_with_path(self, data: Union[torch.Tensor, np.ndarray], path: str,
                       meta_data: Optional[Dict] = None) -> None:
        """
        Save data into a Nifti file.
        The meta_data could optionally have the following keys:

            - ``'filename_or_obj'`` -- for output file name creation, corresponding to filename or object.
            - ``'original_affine'`` -- for data orientation handling, defaulting to an identity matrix.
            - ``'affine'`` -- for data output affine, defaulting to an identity matrix.
            - ``'spatial_shape'`` -- for data output shape.
            - ``'patch_index'`` -- if the data is a patch of big image, append the patch index to filename.

        When meta_data is specified, the saver will try to resample batch data from the space
        defined by "affine" to the space defined by "original_affine".

        If meta_data is None, use the default index (starting from 0) as the filename.

        Args:
            data: target data content that to be saved as a NIfTI format file.
                Assuming the data shape starts with a channel dimension and followed by spatial dimensions.
            meta_data: the meta data information corresponding to the data.
            path: folder path where to store the NIFTI volume

        See Also
            :py:meth:`monai.data.nifti_writer.write_nifti`
        """

        original_affine = meta_data.get("original_affine", None) if meta_data else None
        affine = meta_data.get("affine", None) if meta_data else None
        spatial_shape = meta_data.get("spatial_shape", None) if meta_data else None

        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()

        # change data shape to be (channel, h, w, d)
        while len(data.shape) < 4:
            data = np.expand_dims(data, -1)
        # change data to "channel last" format and write to nifti format file
        data = np.moveaxis(np.asarray(data), 0, -1)

        # if desired, remove trailing singleton dimensions
        if self.squeeze_end_dims:
            while data.shape[-1] == 1:
                data = np.squeeze(data, -1)

        write_nifti(
            data,
            file_name=path,
            affine=affine,
            target_affine=original_affine,
            resample=self.resample,
            output_spatial_shape=spatial_shape,
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
            dtype=self.dtype,
            output_dtype=self.output_dtype,
        )

        if self.print_log:
            print(f"file written: {path}.")
