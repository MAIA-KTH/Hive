import nibabel as nib
from scipy.ndimage import label


def semantic_segmentation_to_instance(mask_filename: str, output_path: str) -> int:
    """
        Given a semantic segmentation mask convert to instance segmentation and save in the given output path.
        Return the number of labels in instance segmentation mask.

        Parameters
        ----------
        mask_filename:
            File path of semantic segmentation mask.
        output_path:
            Output path including new instance segmentation mask file name.

        Returns
        -------
            Number of labels in converted instance segmentation mask.
        """

    # load segmentation mask and properties
    mask = nib.load(mask_filename)
    affine = mask.affine
    np_mask = mask.get_fdata()

    # label connected regions in segmentation mask
    labeled_array, num_features = label(np_mask)

    thresh = 10
    # voxel count in each region from https://neurostars.org/t/roi-voxel-count-using-python/6451
    # ignore regions below threshold = 10
    for i in range(1, num_features + 1):
        vox_count = (labeled_array == i).sum()
        if vox_count < thresh:
            labeled_array[labeled_array == i] = 0

    labeled_array[labeled_array>0] = 1
    labeled_array, num_features = label(labeled_array)

    # convert labeled array into Nifti file
    labeled_mask = nib.Nifti1Image(labeled_array, affine=affine)
    nib.save(labeled_mask, output_path)
    return num_features
