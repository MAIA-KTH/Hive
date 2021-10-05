import math
from os import PathLike
from typing import Union, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from monai.transforms import LoadImaged, Compose, AsChannelFirstD
from nptyping import NDArray
from scipy.ndimage import center_of_mass

from Hive.monai.transforms import OrientToRAId
from Hive.utils.volume_utils import apply_affine_transform_to_vector_field

VECTOR_COORDINATES_2D = {"axial": [0, 1], "coronal": [2, 0], "sagittal": [2, 1]}


def create_2D_GIF_for_vector_field(
    image_filename: Union[str, PathLike],
    vector_field_filename: Union[str, PathLike],
    orientation: str,
    output_filename: Union[str, PathLike],
    step: int = 10,
    grid_spacing: int = 100,
    mask_filename: Union[str, PathLike] = None,
):
    """
    For a given 3D volume and its corresponding vector field, creates an animated GIF, slicing in 2D planes along the
    specified orientation.
    The vector field is visualized as the vector sum, computed according to a specified grid spacing or, optionally,
    according to the labels specified in the mask file.

    Parameters
    ----------
    image_filename : Union[str, PathLike]
        Image file path.
    vector_field_filename : Union[str, PathLike]
        4D Vector field file path.
    orientation : str
        Slicing orientation. Can be 'axial', 'coronal' or 'sagittal'.
    output_filename : Union[str, PathLike]
        Output GIF filename.
    step : int
        Step length used for GIF animation.
    grid_spacing : int
        Grid spacing used in vector field sum.
    mask_filename : Union[str, PathLike]
        Optional mask file path.
    """
    data = {"image": image_filename, "vector_field": vector_field_filename}

    if mask_filename is not None:
        data["mask"] = mask_filename

    transform = Compose(
        [
            LoadImaged(keys=list(data.keys())),
            OrientToRAId(keys=list(data.keys()), slicing_axes=orientation),
            AsChannelFirstD(keys=["vector_field"]),
        ]
    )

    data = transform(data)

    affine_transform = np.eye(4)
    affine_transform[:3, :3] = data["image_meta_dict"]["rotation_affine"]

    data["vector_field"] = apply_affine_transform_to_vector_field(data["vector_field"], affine_transform)

    fig, ax = plt.subplots()

    def update(frame):
        if mask_filename is not None:
            vector_slice_array = sum_vector_field_with_mask(
                data["vector_field"][VECTOR_COORDINATES_2D[orientation], frame, :, :], data["mask"][frame, :, :]
            )
        else:
            vector_slice_array = sum_vector_field(
                data["vector_field"][VECTOR_COORDINATES_2D[orientation], frame, :, :], grid_spacing
            )

        plt.clf()
        plt.axis("off")
        plt.imshow(data["image"][frame, :, :], cmap="gray", origin="lower")
        if mask_filename is not None:
            plt.imshow(
                np.ma.masked_where(data["mask"][frame, :, :] < 0.5, data["mask"][frame, :, :]),
                cmap="gray",
                alpha=0.6,
                origin="lower",
            )

        for i in range(data["vector_field"].shape[2]):
            for j in range(data["vector_field"].shape[3]):
                if vector_slice_array[0, i, j] != 0 or vector_slice_array[1, i, j] != 0:
                    plt.arrow(
                        j,
                        i,
                        vector_slice_array[0, i, j],
                        vector_slice_array[1, i, j],
                        head_width=10,
                        head_length=10,
                        fc="red",
                        ec="red",
                    )

    anim = FuncAnimation(fig, update, frames=np.arange(0, data["vector_field"].shape[1], step), interval=200)
    anim.save(output_filename, dpi=80, writer="imagemagick")


def sum_vector_field(vector_array: NDArray[(3, Any, Any, Any), float], spacing: int) -> NDArray[(3, Any, Any, Any)]:
    """
    Sum a vector field according to the specified 2D grid and set the vector sum origin to the Center of Mass.

    Parameters
    ----------
    vector_array : NDArray[(3, Any, Any, Any), float]
        4D vector field.
    spacing : int
        Spacing for the grid used in the vector sum.

    Returns
    -------
    NDArray[(3, Any, Any, Any), float]
        Vector field with vector sum over the specified 2D grid.
    """
    output_field = np.zeros(vector_array.shape)

    for i in range(0, output_field.shape[1], spacing):
        for j in range(0, output_field.shape[2], spacing):
            end_j = j + spacing
            if end_j > output_field.shape[2]:
                end_j = output_field.shape[2]
            end_i = i + spacing
            if end_i > output_field.shape[1]:
                end_i = output_field.shape[1]

            grid_vector = vector_array[:, i:end_i, j:end_j]
            grid_vector = np.sum(grid_vector ** 2, axis=0)
            grid_vector = np.sqrt(grid_vector)
            mask_vector = np.where(grid_vector > 0, 1, 0)
            cm = center_of_mass(grid_vector, mask_vector)
            if not math.isnan(cm[0]) and not math.isnan(cm[1]):
                output_field[0, i + int(cm[0]), j + int(cm[1])] = np.sum(vector_array[0, i:end_i, j:end_j])
                output_field[1, i + int(cm[0]), j + int(cm[1])] = np.sum(vector_array[1, i:end_i, j:end_j])

    return output_field


def sum_vector_field_with_mask(
    vector_array: NDArray[(3, Any, Any, Any)], mask_array: NDArray[(Any, Any, Any)]
) -> NDArray[(3, Any, Any, Any)]:
    """
    Sum a vector field according to the given labels and set the vector sum origin to the Center of Mass.

    Parameters
    ----------
    vector_array : NDArray[(3, Any, Any, Any)]
        4D vector field.
    mask_array : NDArray[( Any, Any, Any)]
        3D mask file path.

    Returns
    -------
    NDArray[(3, Any, Any, Any), float]
        Vector field with vector sum over the specified labels.
    """
    output_field = np.zeros(vector_array.shape)
    labels = list(np.unique(mask_array))
    labels.remove(0)
    for label in labels:
        vector_field = vector_array * (mask_array == label)

        grid_vector = np.sum(vector_field ** 2, axis=0)
        grid_vector = np.sqrt(grid_vector)
        mask_vector = np.where(grid_vector > 0, 1, 0)
        cm = center_of_mass(grid_vector, mask_vector)

        if not math.isnan(cm[0]) and not math.isnan(cm[1]):
            output_field[0, int(cm[0]), int(cm[1])] = np.sum(vector_field[0, :])
            output_field[1, int(cm[0]), int(cm[1])] = np.sum(vector_field[1, :])
    return output_field
