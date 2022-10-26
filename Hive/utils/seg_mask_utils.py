import numpy as np
import nibabel as nib
import os
from os import PathLike



def semantic_segmentation_to_instance(mask_filename, output):
    """

    Parameters
    ----------
    mask_filename
    output
    """

    mask = nib.load(mask_filename)
    print("Shape of example image: ", mask.shape)
    #convert to numpyarray
    mask.get_data_dtype() == np.dtype(np.int16)
