import math
from os import PathLike
from typing import Union, Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from Hive.monai.transforms import OrientToRAId
from Hive.utils.df_utils import unflatten_dataframe
from Hive.utils.volume_utils import apply_affine_transform_to_vector_field
from matplotlib.animation import FuncAnimation
from monai.transforms import LoadImaged, Compose, AsChannelFirstD
from nptyping import NDArray
from pandas import DataFrame
from scipy import stats
from scipy.ndimage import center_of_mass
from statsmodels.multivariate.manova import MANOVA


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


def create_plotly_3D_vector_field(
    subject_ID: str,
    image_filename: Union[str, PathLike],
    case_summary_filename: Union[str, PathLike],
    output_filename: Union[str, PathLike],
    label_dict: Dict[str, str],
):
    """
    For a given Subject vector field summary, creates and saves the HTML plotly plot of the vector field.

    Parameters
    ----------
    subject_ID : str
        Subject ID.
    image_filename : Union[str, PathLike]
        Image file path, used to render the volume in the HTML Plotly output.
    case_summary_filename : Union[str, PathLike]
        File path of the subject vector field summary.
    output_filename :  Union[str, PathLike]
        HTML output file path.
    label_dict : Dict[str, str]
        Dictionary mapping the label values to their corresponding name.
    """
    vector_scale = 50
    data = {"image": image_filename}

    transform = Compose(
        [
            LoadImaged(keys=["image"]),
            OrientToRAId(keys=["image"], slicing_axes="sagittal"),
        ]
    )

    data = transform(data)
    volume = data["image"]
    volume = np.transpose(volume, (2, 1, 0))
    r, c = volume[0].shape

    nb_frames = volume.shape[0]

    fig = go.Figure(
        frames=[
            go.Frame(
                data=go.Surface(z=k * np.ones((r, c)), surfacecolor=volume[k], cmin=-1000, cmax=1000),
                name=str(k),  # you need to name the frame for the animation to behave properly
            )
            for k in range(0, nb_frames, 50)
        ]
    )

    # Add data to be displayed before animation starts
    fig.add_trace(
        go.Surface(
            z=0 * np.ones((r, c)),
            surfacecolor=volume[0],
            colorscale="Gray",
            cmin=-1000,
            cmax=1000,
            showscale=False,
            colorbar=dict(thickness=20, ticklen=4),
        )
    )

    def frame_args(duration):
        return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

    sliders = [
        {
            "pad": {"b": 10, "t": 60},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [[f.name], frame_args(0)],
                    "label": f.name,
                    "method": "animate",
                }
                for k, f in enumerate(fig.frames)
            ],
        }
    ]

    # Layout
    fig.update_layout(
        width=1440,
        height=810,
        scene=dict(
            zaxis=dict(range=[-1, nb_frames - 1], autorange=False),
            aspectratio=dict(x=1, y=1, z=1),
        ),
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, frame_args(50)],
                        "label": "&#9654;",  # play symbol
                        "method": "animate",
                    },
                    {
                        "args": [[None], frame_args(0)],
                        "label": "&#9724;",  # pause symbol
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "type": "buttons",
                "x": 0.1,
                "y": 0,
            }
        ],
        sliders=sliders,
    )

    if case_summary_filename.endswith(".xlsx"):
        df = pd.read_excel(case_summary_filename)
    elif case_summary_filename.endswith(".csv"):
        df = pd.read_csv(case_summary_filename)
    elif case_summary_filename.endswith(".pkl"):
        df = pd.read_pickle(case_summary_filename)
    else:
        raise ValueError("Case summary file format not recognized, expected one of: '.xslx', '.csv', '.pkl' ")

    df_copy = df.copy(deep=True)

    key_list = list(label_dict.keys())
    val_list = list(label_dict.values())

    for index, row in df.iterrows():
        row_copy = row.copy(deep=True)
        row_copy["CM_x"] = row["CM_x"] + row["V_x"] / vector_scale
        row_copy["CM_y"] = row["CM_y"] + row["V_y"] / vector_scale
        row_copy["CM_z"] = row["CM_z"] + row["V_z"] / vector_scale
        df_copy = df_copy.append(row_copy, ignore_index=True)

    fig_line3D = px.line_3d(df_copy, x="CM_x", y="CM_y", z="CM_z", color="Label", width=50)
    fig.add_traces(data=fig_line3D.data)
    for index, row in df.iterrows():
        position = val_list.index(row["Label"])

        fig.add_traces(
            data=[
                {
                    "type": "cone",
                    "x": [row["CM_x"] + row["V_x"] / vector_scale],
                    "y": [row["CM_y"] + row["V_y"] / vector_scale],
                    "z": [row["CM_z"] + row["V_z"] / vector_scale],
                    "u": [row["V_x"] / (vector_scale * 3)],
                    "v": [row["V_y"] / (vector_scale * 3)],
                    "w": [row["V_z"] / (vector_scale * 3)],
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
        title="{} LVC Vector Field".format(subject_ID),
        scene={
            "camera": {"eye": {"x": -0.76, "y": 1.8, "z": 0.92}},
            "xaxis": {"title": "x", "nticks": 20, "range": [0, volume.shape[2]]},
            "yaxis": {"title": "y", "nticks": 20, "range": [0, volume.shape[1]]},
            "zaxis": {"title": "z", "nticks": 20, "range": [0, volume.shape[0]]},
        },
    )

    fig.write_html(output_filename)


def create_plotly_3D_vector_field_summary(
    summary_filename: Union[str, PathLike], output_filename: Union[str, PathLike], dataset_name: str
):
    """
    For a given Dataset vector field summary, creates and saves the HTML plotly plot of the vector field.

    Parameters
    ----------
    summary_filename : Union[str, PathLike]
        File path of the dataset vector field summary.
    output_filename : Union[str, PathLike]
        HTML output file path.
    dataset_name : str
        Dataset string name.
    """
    if summary_filename.endswith(".xlsx"):
        df = pd.read_excel(summary_filename)
    elif summary_filename.endswith(".csv"):
        df = pd.read_csv(summary_filename)
    elif summary_filename.endswith(".pkl"):
        df = pd.read_pickle(summary_filename)
    else:
        raise ValueError("Summary file format not recognized, expected one of: '.xslx', '.csv', '.pkl' ")

    for index, row in df.iterrows():
        magnitude = math.sqrt(math.pow(row["V_x"], 2) + math.pow(row["V_y"], 2) + math.pow(row["V_z"], 2))
        df.at[index, "V_x"] = row["V_x"] / magnitude
        df.at[index, "V_y"] = row["V_y"] / magnitude
        df.at[index, "V_z"] = row["V_z"] / magnitude

    fig = px.scatter_3d(df, x="V_x", y="V_y", z="V_z", color="Label", hover_data=["Subject"])

    fig.update_layout(title="{} LVC Vector Field Summary".format(dataset_name))
    fig.write_html(output_filename)


def convert_cartesian_to_spherical_coordinates(
    x: NDArray[(Any,), float], y: NDArray[(Any,), float], z: NDArray[(Any,), float]
) -> Tuple[NDArray[(Any,), float], NDArray[(Any,), float], NDArray[(Any,), float]]:
    """
    Converts (x,y,z) Numpy arrays of cartesian coordinates into (r, elevationn, azimuth) Numpy arrays of the corresponding
    spherical coordinates.

    Parameters
    ----------
    x : NDArray[(Any,), float]
        X Cartesian coordinates array
    y : NDArray[(Any,), float]
        Y Cartesian coordinates array
    z : NDArray[(Any,), float]
        Z Cartesian coordinates array

    Returns
    -------
    Tuple[NDArray[(Any,), float], NDArray[(Any, ), float], NDArray[(Any,), float]]
        Tuple of 3 Numpy array, including the spherical coordinates (r, elevationn, azimuth).
    """
    XsqPlusYsq = x ** 2 + y ** 2
    r = np.sqrt(XsqPlusYsq + z ** 2)  # r
    elev = np.arctan2(z, np.sqrt(XsqPlusYsq))  # theta
    az = np.arctan2(y, x)  # phi

    return r, elev, az


def get_sphericaL_coordinates_df(df_filepath: Union[str, PathLike], labels: List[str]) -> DataFrame:
    """
    Load a DataFrame and converts all the coordinates in the ``V_x_LABEL``, ``V_y_LABEL`` and ``V_z_LABEL`` from
    cartesian to spherical coordinates, for each LABEL included in ``labels``.
    Returns a DataFrame with spherical coordinates for each Subject and each Label.

    Parameters
    ----------
    df_filepath : Union[str, PathLike]
        DataFrame file path.
    labels  : List[str]
        List of label to be transformed.

    Returns
    -------
    DataFrame
        Pandas DataFrame with Spherical coordinates for the given values, for each label.
    """
    if df_filepath.endswith(".xlsx"):
        df = pd.read_excel(df_filepath)
    elif df_filepath.endswith(".csv"):
        df = pd.read_csv(df_filepath)
    elif df_filepath.endswith(".pkl"):
        df = pd.read_pickle(df_filepath)
    else:
        raise ValueError("Case summary file format not recognized, expected one of: '.xslx', '.csv', '.pkl' ")

    df_vectors = unflatten_dataframe(df, "Label", "Subject", ["V_x", "V_y", "V_z"])
    subject_list = df_vectors["Subject"].values
    spherical_coord_list = []
    for label in labels:
        r, theta, phi = convert_cartesian_to_spherical_coordinates(
            df_vectors["V_x_{}".format(label)].values,
            df_vectors["V_y_{}".format(label)].values,
            df_vectors["V_z_{}".format(label)].values,
        )
        spherical_array_coord = np.vstack((theta, phi)).T
        for spherical_coord, subject in zip(spherical_array_coord, subject_list):
            spherical_coord_dict = {
                "Subject": subject,
                "Label": label,
                "Elevation": spherical_coord[0],
                "Azimuth": spherical_coord[1],
            }
            spherical_coord_list.append(spherical_coord_dict)
    spherical_coord_df = pd.DataFrame(spherical_coord_list)
    return spherical_coord_df


def calculate_z_score(df: DataFrame, labels: List[str], columns: List[str]) -> DataFrame:
    """
    Calculates Z scores on the given columns of the DataFrame, for the given labels, and drops the rows where the Z score
    is > 2.5 .
    Returns the reduced DataFrame.

    Parameters
    ----------
    df  : DataFrame
        Input DataFrame
    labels : List[str]
        List of labels.
    columns : List[str]
        List of columns to compute the Z score on.

    Returns
    -------
    DataFrame
        Pandas DataFrame, where rows with Z score > 2.5 are removed.
    """
    z_scores = {}
    df_corrected = df.copy(deep=True)
    for label in labels:
        z_scores[label] = []
        df_label = df[df["Label"] == label]
        df_label_indexes = df.index[df["Label"] == label].tolist()
        for column in columns:
            z_scores[label].append((np.abs(stats.zscore(df_label[column].values)) > 2.5).astype(np.uint8))
        z_scores[label] = np.sum(z_scores[label], axis=0)
        for idx, z_score in enumerate(z_scores[label]):
            if z_score != 0:
                df_corrected.drop(labels=df_label_indexes[idx], axis=0, inplace=True)

    return df_corrected


def plot_spherical_coordinates(
    df_spherical: DataFrame, output_spherical_plot: Union[str, PathLike], dataset_name: str, label_dict: Dict[str, str]
) -> DataFrame:
    """
    For a given DataFrame with spherical coordinates, generates the corresponding 2D scatter plot (elevation. azimuth),
    and return a MANOVA DataFrame, including the MANOVA analysis: ``'Elevation + Azimuth ~ Label'``.

    Parameters
    ----------
    df_spherical : DataFrame
        DataFrame with Spherical coordinates.
    output_spherical_plot : Union[str, PathLike]
        Output file path to save the scatter 2D plot.
    dataset_name : str
        Dataset string name.
    label_dict : Dict[str, str]
        Dictionary mapping the label values to their corresponding name.

    Returns
    -------
    DataFrame
        MANOVA DataFrame
    """
    mapping = {label_dict[label]: int(label) for label in label_dict if label != "0"}

    key_list = list(label_dict.keys())
    val_list = list(label_dict.values())

    fig = go.Figure()
    fig.add_traces(data=px.scatter(df_spherical, x="Elevation", y="Azimuth", color="Label", hover_data=["Subject"]).data)

    labels = df_spherical["Label"].unique()
    df_spherical_z_corrected = calculate_z_score(df_spherical, labels, ["Elevation", "Azimuth"])
    label_mean = df_spherical_z_corrected.groupby(["Label"]).mean()
    label_sd = df_spherical_z_corrected.groupby(["Label"]).std()
    for label in labels:
        position = val_list.index(label)
        min_x = label_mean["Elevation"][label] - 3 * label_sd["Elevation"][label]
        max_x = label_mean["Elevation"][label] + 3 * label_sd["Elevation"][label]
        min_y = label_mean["Azimuth"][label] - 3 * label_sd["Azimuth"][label]
        max_y = label_mean["Azimuth"][label] + 3 * label_sd["Azimuth"][label]
        fig.add_shape(
            type="circle",
            xref="x",
            yref="y",
            x0=min_x,
            y0=min_y,
            x1=max_x,
            y1=max_y,
            opacity=0.2,
            fillcolor=px.colors.qualitative.Plotly[int(key_list[position]) - 1],
            line_color=px.colors.qualitative.Plotly[int(key_list[position]) - 1],
        )
    df_spherical = df_spherical.replace({"Label": mapping})
    df_spherical.drop(labels="Subject", axis=1, inplace=True)
    maov = MANOVA.from_formula("Elevation + Azimuth ~ Label", data=df_spherical)
    manova_df = maov.mv_test().summary_frame
    fig.update_layout(
        title="{} Spherical LVC Vector Field, MANOVA p_value: {}".format(
            dataset_name, manova_df["Pr > F"]["Label"]["Wilks' lambda"]
        ),
        xaxis_title="Elevation",
        yaxis_title="Azimuth",
    )
    fig.write_html(output_spherical_plot)

    return manova_df
