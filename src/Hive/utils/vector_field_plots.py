import math
from os import PathLike
from typing import Union, Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from Hive.monai.transforms import OrientToRAId
from Hive.utils.volume_utils import apply_affine_transform_to_vector_field
from matplotlib.animation import FuncAnimation
from monai.transforms import LoadImaged, Compose, AsChannelFirstD
from nptyping import NDArray
from scipy.ndimage import center_of_mass


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
    axis_orientation = data["image_meta_dict"]["axis_orientation_swapped"]
    vector_field_2D_axes = axis_orientation[1:]
    plot_origin = "lower"

    if orientation != "axial":
        plot_origin = "upper"
        axis_orientation.reverse()
        vector_field_2D_axes = axis_orientation[:2]
        data["image"] = np.swapaxes(data["image"], 1, 2)
        data["vector_field"] = np.swapaxes(data["vector_field"], 2, 3)
        if "mask" in data:
            data["mask"] = np.swapaxes(data["mask"], 1, 2)

    affine_transform = np.eye(4)
    affine_transform[:3, :3] = np.transpose(data["image_meta_dict"]["rotation_affine"], (1, 0))

    data["vector_field"] = apply_affine_transform_to_vector_field(data["vector_field"], affine_transform)

    fig, ax = plt.subplots()

    def update(frame):
        if mask_filename is not None:
            vector_slice_array = sum_vector_field_with_mask(
                data["vector_field"][vector_field_2D_axes, frame, :, :], data["mask"][frame, :, :]
            )
        else:
            vector_slice_array = sum_vector_field(data["vector_field"][vector_field_2D_axes, frame, :, :], grid_spacing)

        plt.clf()
        plt.axis("off")
        plt.imshow(data["image"][frame, :, :], cmap="gray", origin=plot_origin)
        if mask_filename is not None:
            plt.imshow(
                np.ma.masked_where(data["mask"][frame, :, :] < 0.5, data["mask"][frame, :, :]),
                cmap="gray",
                alpha=0.6,
                origin=plot_origin,
            )

        for i in range(data["vector_field"].shape[2]):
            for j in range(data["vector_field"].shape[3]):
                if vector_slice_array[0, i, j] != 0 or vector_slice_array[1, i, j] != 0:
                    x_pos = i
                    y_pos = j
                    if orientation == "axial":
                        x_pos = j
                        y_pos = i
                    plt.arrow(
                        x_pos,
                        y_pos,
                        vector_slice_array[0, i, j],
                        vector_slice_array[1, i, j],
                        head_width=10,
                        head_length=10,
                        fc="red",
                        ec="red",
                    )

    anim = FuncAnimation(fig, update, frames=np.arange(0, data["vector_field"].shape[1], step), interval=200)
    anim.save(output_filename, dpi=80, writer="imagemagick")


def sum_vector_field(vector_array: NDArray[(2, Any, Any), float], spacing: int) -> NDArray[(2, Any, Any), float]:
    """
    Sum a vector field according to the specified 2D grid and set the vector sum origin to the Center of Mass.

    Parameters
    ----------
    vector_array : NDArray[(2, Any, Any), float]
        3D vector field.
    spacing : int
        Spacing for the grid used in the vector sum.

    Returns
    -------
    NDArray[(2, Any, Any), float]
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
    vector_array: NDArray[(2, Any, Any), float], mask_array: NDArray[(Any, Any), int]
) -> NDArray[(2, Any, Any), float]:
    """
    Sum a vector field according to the given labels and set the vector sum origin to the Center of Mass.

    Parameters
    ----------
    vector_array : NDArray[(2, Any, Any), float]
        3D vector field.
    mask_array : NDArray[(Any, Any), int]
        2D mask file path.

    Returns
    -------
    NDArray[(2, Any, Any), float]
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


def sum_vector_field_3D(vector_array: NDArray[(3, Any, Any, Any), float], spacing: int) -> NDArray[(3, Any, Any, Any), float]:
    """
    Sum a vector field according to the specified 3D grid and set the vector sum origin to the Center of Mass.

    Parameters
    ----------
    vector_array : NDArray[(3, Any, Any, Any), float]
        4D vector field.
    spacing : int
        Spacing for the grid used in the vector sum.

    Returns
    -------
    NDArray[(3, Any, Any, Any), float]
        Vector field with vector sum over the specified 3D grid.
    """
    output_field = np.zeros(vector_array.shape)

    for i in range(0, output_field.shape[1], spacing):
        for j in range(0, output_field.shape[2], spacing):
            for k in range(0, output_field.shape[3], spacing):
                end_j = j + spacing
                if end_j > output_field.shape[2]:
                    end_j = output_field.shape[2]
                end_i = i + spacing
                if end_i > output_field.shape[1]:
                    end_i = output_field.shape[1]
                end_k = k + spacing
                if end_k > output_field.shape[3]:
                    end_k = output_field.shape[3]

                grid_vector = vector_array[:, i:end_i, j:end_j, k:end_k]
                grid_vector = np.sum(grid_vector ** 2, axis=0)
                grid_vector = np.sqrt(grid_vector)
                mask_vector = np.where(grid_vector > 0, 1, 0)
                cm = center_of_mass(grid_vector, mask_vector)
                if not math.isnan(cm[0]) and not math.isnan(cm[1]) and not math.isnan(cm[2]):
                    output_field[0, i + int(cm[0]), j + int(cm[1]), k + int(cm[2])] = np.sum(
                        vector_array[0, i:end_i, j:end_j, k:end_k]
                    )
                    output_field[1, i + int(cm[0]), j + int(cm[1]), k + int(cm[2])] = np.sum(
                        vector_array[1, i:end_i, j:end_j, k:end_k]
                    )
                    output_field[2, i + int(cm[0]), j + int(cm[1]), k + int(cm[2])] = np.sum(
                        vector_array[2, i:end_i, j:end_j, k:end_k]
                    )

    return output_field


def sum_vector_field_3D_with_mask(
    vector_array: NDArray[(3, Any, Any, Any), float], mask_array: NDArray[(3, Any, Any, Any), int]
) -> (NDArray[(3, Any, Any, Any), float], Dict[int, Any]):
    """
    Sum a vector field according to the given labels and set the vector sum origin to the Center of Mass.

    Parameters
    ----------
    vector_array : NDArray[(3, Any, Any, Any), float]
        4D vector field.
    mask_array : NDArray[(Any, Any, Any), int]
        3D mask file path.

    Returns
    -------
    NDArray[(3, Any, Any, Any), float]
        Vector field with vector sum over the specified labels.
    """
    output_field = np.zeros(vector_array.shape)
    voxel_cm_dict = {}
    labels = list(np.unique(mask_array))
    labels.remove(0)
    for label in labels:
        vector_field = vector_array * (mask_array == label)

        grid_vector = np.sum(vector_field ** 2, axis=0)
        grid_vector = np.sqrt(grid_vector)
        mask_vector = np.where(grid_vector > 0, 1, 0)
        cm = center_of_mass(grid_vector, mask_vector)

        if not math.isnan(cm[0]) and not math.isnan(cm[1]) and not math.isnan(cm[2]):
            output_field[0, int(cm[0]), int(cm[1]), int(cm[2])] = np.sum(vector_field[0, :])
            output_field[1, int(cm[0]), int(cm[1]), int(cm[2])] = np.sum(vector_field[1, :])
            output_field[2, int(cm[0]), int(cm[1]), int(cm[2])] = np.sum(vector_field[2, :])
            voxel_cm_dict[int(label)] = [int(cm[0]), int(cm[1]), int(cm[2])]
    return output_field, voxel_cm_dict


def get_vector_field_summary_for_case(
    image_filename: Union[str, PathLike],
    vector_field_filename: Union[str, PathLike],
    label_filename: Union[str, PathLike],
    case_id: str,
    label_dict: Dict[str, str],
) -> List[Dict[str, Any]]:
    """
    For a given case ( including image volume, vector field map and label mask), returns the vector field summary,
    specifying for each label the vector sum and its Center of Mass.

    Parameters
    ----------
    image_filename : Union[str, PathLike]
        Image volume filename.
    vector_field_filename : Union[str, PathLike]
        Vector field filename.
    label_filename : Union[str, PathLike]
        Label mask filename.
    case_id : str
        Case ID Code.
    label_dict : Dict[str, str]
        Dictionary mapping each label number to its corresponding anatomic area.

    Returns
    -------
    List[Dict[str, Any]]
        List of dictionaries ( one element per label ), containing vector field geometrical information.
    """
    data = {"image": image_filename, "vector_field": vector_field_filename}

    if label_filename is not None:
        data["label"] = label_filename

    transform = Compose(
        [
            LoadImaged(keys=list(data.keys())),
            OrientToRAId(keys=list(data.keys())),
            AsChannelFirstD(keys=["vector_field"]),
        ]
    )

    data = transform(data)

    affine_transform = np.eye(4)
    affine_transform[:3, :3] = np.transpose(data["image_meta_dict"]["rotation_affine"], (1, 0))

    data["vector_field"] = apply_affine_transform_to_vector_field(data["vector_field"], affine_transform)

    vector_array, voxel_cm_dict = sum_vector_field_3D_with_mask(data["vector_field"], data["label"])

    case_vector_field_summary = []

    for label in voxel_cm_dict:
        case_vector_field_summary_label = {
            "Subject": case_id,
            "Label": label_dict[str(label)],
            "CM_x": voxel_cm_dict[label][0],
            "CM_y": voxel_cm_dict[label][1],
            "CM_z": voxel_cm_dict[label][2],
            "V_x": vector_array[0, voxel_cm_dict[label][0], voxel_cm_dict[label][1], voxel_cm_dict[label][2]],
            "V_y": vector_array[1, voxel_cm_dict[label][0], voxel_cm_dict[label][1], voxel_cm_dict[label][2]],
            "V_z": vector_array[2, voxel_cm_dict[label][0], voxel_cm_dict[label][1], voxel_cm_dict[label][2]],
        }
        case_vector_field_summary.append(case_vector_field_summary_label)

    return case_vector_field_summary


def create_plotly_3D_vector_field(case_summary_filename: Union[str, PathLike], output_filename: Union[str, PathLike]):
    """
    For a given Subject vector field summary, creates and saves the HTML plotly plot of the vector field.

    Parameters
    ----------
    case_summary_filename : Union[str, PathLike]
        File path of the subject vector field summary.
    output_filename : : Union[str, PathLike]
        HTML output file path.
    """
    df = pd.read_excel(case_summary_filename)

    df_copy = df.copy(deep=True)
    label_dict = {"0": "Background", "1": "LU", "2": "LL", "3": "RU", "4": "RM", "5": "RL"}

    key_list = list(label_dict.keys())
    val_list = list(label_dict.values())

    for index, row in df.iterrows():
        row_copy = row.copy(deep=True)
        row_copy["CM_x"] = row["CM_x"] + row["V_x"] / 100
        row_copy["CM_y"] = row["CM_y"] + row["V_y"] / 100
        row_copy["CM_z"] = row["CM_z"] + row["V_z"] / 100
        df_copy = df_copy.append(row_copy, ignore_index=True)

    fig = px.line_3d(df_copy, x="CM_x", y="CM_y", z="CM_z", color="Label")
    for index, row in df.iterrows():
        position = val_list.index(row["Label"])

        fig.add_traces(
            data=[
                {
                    "type": "cone",
                    "x": [row["CM_x"] + row["V_x"] / 100],
                    "y": [row["CM_y"] + row["V_y"] / 100],
                    "z": [row["CM_z"] + row["V_z"] / 100],
                    "u": [row["V_x"] / 300],
                    "v": [row["V_y"] / 300],
                    "w": [row["V_z"] / 300],
                    "colorscale": [
                        [0, px.colors.qualitative.Plotly[int(key_list[position]) - 1]],
                        [1, px.colors.qualitative.Plotly[int(key_list[position]) - 1]],
                    ],
                    "showscale": False,
                    "showlegend": True,
                    "name": row["Label"],
                }
            ]
        )

    fig.update_layout(
        title="LVC Vector Field",
        scene={
            "camera": {"eye": {"x": -0.76, "y": 1.8, "z": 0.92}},
            "xaxis": {"title": "x", "nticks": 20, "range": [0, 500]},
            "yaxis": {"title": "y", "nticks": 20, "range": [0, 500]},
            "zaxis": {"title": "z", "nticks": 20, "range": [0, 500]},
        },
    )

    fig.write_html(output_filename)


def create_plotly_3D_vector_field_summary(summary_filename: Union[str, PathLike], output_filename: Union[str, PathLike]):
    """
    For a given Dataset vector field summary, creates and saves the HTML plotly plot of the vector field.

    Parameters
    ----------
    summary_filename : Union[str, PathLike]
        File path of the dataset vector field summary.
    output_filename : : Union[str, PathLike]
        HTML output file path.
    """
    df = pd.read_excel(summary_filename)

    for index, row in df.iterrows():
        magnitude = math.sqrt(math.pow(row["V_x"], 2) + math.pow(row["V_y"], 2) + math.pow(row["V_z"], 2))
        df.at[index, "V_x"] = row["V_x"] / magnitude
        df.at[index, "V_y"] = row["V_y"] / magnitude
        df.at[index, "V_z"] = row["V_z"] / magnitude

    fig = px.scatter_3d(df, x="V_x", y="V_y", z="V_z", color="Label", hover_data=["Subject"])

    fig.update_layout(title="LVC Vector Field Summary")
    fig.write_html(output_filename)
