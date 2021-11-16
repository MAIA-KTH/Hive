import copy
import json
from os import PathLike
from pathlib import Path
from typing import List, Dict, Any, Union

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from Hive.evaluation import METRICS_FOLDER_NAME
from Hive.utils.log_utils import get_logger, DEBUG
from pandas import DataFrame
from plotly.graph_objects import Figure

logger = get_logger(__name__)

ADVANCED_METRICS = {
    "Specificity": {
        "Base_Metrics": ["False Positive Rate"],
        "Function": lambda x: 1 - x,
    },
    "Fowlkes–Mallows index": {"Base_Metrics": ["Precision", "Recall"], "Function": lambda x, y: np.sqrt(np.multiply(x, y))},
    "Relative Volumetric Difference": {
        "Base_Metrics": ["Total Positives Reference", "Total Positives Test"],
        "Function": lambda x, y: np.divide(np.subtract(y, x), x) * 100,
    },
    "Relative Absolute Volumetric Difference": {
        "Base_Metrics": ["Total Positives Reference", "Total Positives Test"],
        "Function": lambda x, y: np.divide(np.abs(np.subtract(y, x)), x) * 100,
    },
}  # type: Dict[str, Any]


def get_subject_table(subject_json_file: Union[str, PathLike]) -> Figure:
    """
    Creates and returns a ```Plotly`` :py:class`go.Table` including Subject IDs, according to the given JSON file.

    Parameters
    ----------
    subject_json_file : Union[str, PathLike]
        JSON file path, including dict for Subject IDs.

    Returns
    -------
    Figure
        ```Plotly`` :py:class`go.Table`.
    """
    df = pd.DataFrame(columns=["Subject", "ID"])
    with open(subject_json_file) as json_file:
        phase_dict = json.load(json_file)

    for key in phase_dict:
        df = df.append({"Subject": key, "ID": phase_dict[key]}, ignore_index=True)

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=["Subject", "ID"], fill_color="paleturquoise", align="left"),
                cells=dict(values=[df.Subject, df.ID], fill_color="lavender", align="left"),
            )
        ]
    )
    fig.update_layout(title="Subject IDs")
    return fig


def get_results_summary_filepath(config_dict: Dict[str, Any], section: str, result_suffix: str, fold: int = 0) -> str:
    """
     Return the JSON results filepath, for a specified section ( ``validation`` or ``testing`` ), and/or fold number.

     Parameters
     ----------
    config_dict : Dict[str, Any]
         Configuration dictionary, including experiment settings.
     section : str
         Section name. Values accepted:  ``validation``, ``testing``.
     result_suffix : str
         String used to retrieve the JSON result summaries, from where to extract metric scores.
     fold : int
         Optional fold number, in case section = ``validation``

     Returns
     str
         JSON results summary filepath
     -------

    """

    predictions_path = config_dict["predictions_path"]
    results_folder_name = config_dict["predictions_folder_name"]

    fold_folder = "fold_" + str(fold)
    if section == "validation":
        parent_folder = str(Path(fold_folder).joinpath(results_folder_name))
    elif section == "testing":
        parent_folder = results_folder_name
    else:
        raise ValueError("Invalid section. Expected one of: [ validation, testing ]")
    summary_filepath = str(
        Path(predictions_path).joinpath(
            parent_folder,
            "summary{}.json".format(result_suffix),
        )
    )

    return summary_filepath


def read_metric_list(summary_filepath: Union[str, PathLike], config_dict: Dict[str, Any]) -> List[str]:
    """

    Parameters
    ----------
    summary_filepath : Union[str, PathLike]
        JSON summary filepath.
    config_dict : Dict[str, Any]
        Configuration dictionary, including experiment settings.

    Returns
    -------
    List[str]
        List of metric names
    """
    with open(summary_filepath) as json_file:
        results_summary_data = json.load(json_file)

    if "Cascade" in config_dict and config_dict["Cascade"]:
        label_dict = config_dict["step_{}".format(str(int(config_dict["Cascade_steps"]) - 1))]["label_dict"]
    else:
        label_dict = config_dict["label_dict"]
    label_dict.pop("0", None)
    metric_list = list(results_summary_data["results"]["all"][0][list(label_dict.keys())[0]].keys())

    return metric_list


def save_metrics(
    config_dict: Dict[str, Any],
    metric_name: str,
    basic_metrics: List[str],
    section: str,
    result_suffix: str,
    df_format: str = "pickle",
    subject_phase_dict: Dict[str, str] = None,
):
    """
    For the specified metric and section, saves a Pandas Dataframe, including the class-wise scores and the case IDs.
    If the metric is included in the basic_metrics, read from the JSON summary file, the metrics is automatically written
    in the DataFrame. Otherwise, if the metric is included in the advanced metrics, the corresponding computations are
    performed, before writing the values.
    The Pandas DataFrames are then saved as pickle files.

    Parameters
    ----------
    config_dict : Dict[str, Any]
        Configuration dictionary, including experiment settings.
    metric_name : str
        Metric name.
    section : str
        Section name. Values accepted: ``validation``, ``testing``.
    basic_metrics : List[str]
        List of basic metrics, found in the JSON result summary file
    result_suffix: str
        String used to retrieve the JSON result summaries, from where to extract metrics scores.
    df_format : str
        File format to save the Pandas DataFrame, can be Excel, CSV or Pickle. Defaults to Pickle.
    subject_phase_dict : Dict[str, str]
        Optional Dictionary including breathing Phase for each Subject ID.

    """
    pd.options.display.float_format = "{:.2%}".format

    Path(config_dict["results_folder"]).joinpath(METRICS_FOLDER_NAME, section, metric_name).mkdir(exist_ok=True, parents=True)

    column_id = None

    if "Metrics_save_configs" in config_dict:
        if "ID_column" in config_dict["Metrics_save_configs"]:
            column_id = config_dict["Metrics_save_configs"]["ID_column"]

    if section == "testing":
        n_folds = 1
    elif section == "validation":
        n_folds = config_dict["n_folds"]
    else:
        raise ValueError("Invalid section. Expected one of: [ validation, testing ]")

    if "Cascade" in config_dict and config_dict["Cascade"]:
        label_dict = config_dict["step_{}".format(str(int(config_dict["Cascade_steps"]) - 1))]["label_dict"]
    else:
        label_dict = config_dict["label_dict"]
    label_dict.pop("0", None)

    df = pd.DataFrame()
    subj_id = 0

    composed_metric = None
    if metric_name not in basic_metrics:
        if metric_name in ADVANCED_METRICS:
            composed_metric = metric_name
        else:
            logger.log(DEBUG, "{} is not a valid metric. Skipping.".format(metric_name))
            return
    subject_list = []
    for fold in range(n_folds):

        summary_filepath = get_results_summary_filepath(config_dict, section, result_suffix, fold)
        if not Path(summary_filepath).is_file():
            logger.info("Fold {} not found! Skipping.".format(fold))
            continue
        with open(summary_filepath) as json_file:
            summary_results_data = json.load(json_file)

        for i in range(len(summary_results_data["results"]["all"])):
            df_temp = pd.DataFrame(summary_results_data["results"]["all"][i])
            column_selection = list(label_dict.keys())
            column_rename = copy.deepcopy(label_dict)

            if composed_metric is not None:
                base_metric_0 = ADVANCED_METRICS[composed_metric]["Base_Metrics"][0]
                df_single_temp = (
                    df_temp[column_selection]
                    .loc[[base_metric_0]]
                    .rename(columns=column_rename, index={base_metric_0: str(subj_id)})
                )
                base_metrics = [
                    df_temp[list(label_dict.keys())].loc[[base_metric_i]]
                    for base_metric_i in ADVANCED_METRICS[composed_metric]["Base_Metrics"]
                ]
                df_composed_temp = ADVANCED_METRICS[composed_metric]["Function"](*base_metrics)
                df_single_temp[[label_dict[key] for key in label_dict]] = df_composed_temp.values
            else:
                df_single_temp = (
                    df_temp[column_selection].loc[[metric_name]].rename(columns=column_rename, index={metric_name: str(subj_id)})
                )

            if column_id is not None:
                df_single_temp["ID"] = Path(df_temp[column_id][0]).name[: -len(config_dict["FileExtension"])]
                subject_list.append(df_single_temp["ID"][0])
                if subject_phase_dict is not None:
                    df_single_temp["Phase"] = subject_phase_dict[df_single_temp["ID"][0]]
            if section == "testing":
                df_single_temp["Section"] = "Testing"
            else:
                df_single_temp["Section"] = "Fold {}".format(fold)

            df_single_temp["Experiment"] = config_dict["Experiment Name"]
            df = df.append(df_single_temp)
            subj_id = subj_id + 1

    if composed_metric is not None:
        metric_name = composed_metric

    df_output_path = str(
        Path(config_dict["results_folder"]).joinpath(METRICS_FOLDER_NAME, section, metric_name, "{}".format(metric_name))
    )

    if df_format == "excel":
        writer = pd.ExcelWriter(df_output_path + ".xlsx", engine="xlsxwriter")

    if "Save_stats" in config_dict["Metrics_save_configs"]:
        if config_dict["Metrics_save_configs"]["Save_stats"]:
            df_aggregate = pd.DataFrame(zip(df.mean(), df.std()), columns=["Mean", "SD"], index=label_dict).rename(
                index=label_dict
            )

            if df_format == "pickle":
                df_aggregate.to_pickle(df_output_path + "_stats.pkl")
            elif df_format == "csv":
                df_aggregate.to_csv(df_output_path + "_stats.csv")
            elif df_format == "excel":
                df_aggregate.to_excel(writer, sheet_name="Stats")

    df_flat = df[[label_dict[key] for key in label_dict]].stack()
    df_flat = pd.DataFrame(df_flat)
    df_flat.reset_index(inplace=True)
    df_flat.columns = ["Subject", "Label", metric_name]
    df_flat["Section"] = section.capitalize()
    if subject_phase_dict is not None:
        df_flat["Phase"] = "Not Assigned"
    df_flat["Experiment"] = config_dict["Experiment Name"]

    for index, row in df_flat.iterrows():
        df_flat["Subject"][index] = subject_list[int(row["Subject"])]
        if subject_phase_dict is not None:
            df_flat["Phase"][index] = subject_phase_dict[df_flat["Subject"][index]]
    subject_id = {subject: str(index) for index, subject in enumerate(subject_list)}
    with open(
        Path(config_dict["results_folder"]).joinpath(METRICS_FOLDER_NAME, section, metric_name, "subject_id.json"), "w"
    ) as fp:
        json.dump(subject_id, fp)

    subject_table = get_subject_table(
        str(Path(config_dict["results_folder"]).joinpath(METRICS_FOLDER_NAME, section, metric_name, "subject_id.json"))
    )
    subject_table.write_html(
        str(Path(config_dict["results_folder"]).joinpath(METRICS_FOLDER_NAME, section, metric_name, "subject_id.html"))
    )

    if df_format == "pickle":
        df_flat.to_pickle(df_output_path + "_flat.pkl")
        df.to_pickle(df_output_path + "_table.pkl")
    elif df_format == "csv":
        df_flat.to_csv(df_output_path + "_flat.csv")
        df.to_csv(df_output_path + "_table.csv")
    elif df_format == "excel":
        df_flat.to_excel(writer, sheet_name="Flat", index=False)
        df.to_excel(writer, sheet_name="Table", index=False)
        writer.save()


def create_dataframe_for_project(
    results_folder: Union[str, PathLike],
    project_name: str,
    config_files: List[str],
    metrics: List[str],
    df_format: str = "pickle",
):
    """
    Given a list of experiment configuration files, merges the metric results from each DataFrame in a single DataFrame,
    saving it as **project** section. All the metric DataFrames are stored in a single DataFrame in the *result_folder*,
    with the filename as the *project_name*.

    Parameters
    ----------
    results_folder : Union[str, PathLike]
        Folder to store project metric result DataFrames.
    project_name : str
        String used to save the DataFrame files.
    config_files : List[str]
        List of JSON experiment configuration files, used to retrieve the experiment to include in the project results.
    metrics : List[str]
        List of metrics to include in the DataFrame.
    df_format : str
        File format used to store project DataFrame.
    """
    df_paths = {}
    for config_file in config_files:
        with open(config_file) as json_file:
            config_dict = json.load(json_file)

        df_paths[config_dict["Experiment Name"]] = get_saved_dataframes(config_dict, metrics, ["experiment"], df_format)

    pd_flat_metric_list = []
    pd_metric_list = []

    for metric in metrics:
        for experiment in df_paths.keys():
            df_flat = read_dataframe(df_paths[experiment]["{}_flat_experiment".format(metric)], sheet_name="Flat")
            df_flat = df_flat.rename(columns={metric: "Metric_Score"})
            df_flat["Metric"] = metric
            pd_flat_metric_list.append(df_flat)
            df = read_dataframe(df_paths[experiment]["{}_experiment".format(metric)], sheet_name="Table")
            df["Metric"] = metric
            pd_metric_list.append(df)

    df_flat = pd.concat(pd_flat_metric_list, ignore_index=True)
    df_table = pd.concat(pd_metric_list, ignore_index=True)
    Path(results_folder).joinpath(METRICS_FOLDER_NAME).mkdir(exist_ok=True, parents=True)

    subject_list = set(df_flat["Subject"].tolist())
    subject_id = {subject: str(index) for index, subject in enumerate(subject_list)}
    with open(Path(results_folder).joinpath(METRICS_FOLDER_NAME, "subject_id.json"), "w") as fp:
        json.dump(subject_id, fp)

    subject_table = get_subject_table(str(Path(results_folder).joinpath(METRICS_FOLDER_NAME, "subject_id.json")))
    subject_table.write_html(str(Path(results_folder).joinpath(METRICS_FOLDER_NAME, "subject_id.html")))

    df_file_path = str(Path(results_folder).joinpath(METRICS_FOLDER_NAME, project_name))

    if df_format == "excel":
        writer = pd.ExcelWriter(df_file_path + ".xlsx", engine="xlsxwriter")
        df_table.to_excel(writer, sheet_name="Table", index=False)
        df_flat.to_excel(writer, sheet_name="Flat", index=False)
        writer.save()
    elif df_format == "csv":
        df_table.to_csv(df_file_path + "_table.csv")
        df_flat.to_csv(df_file_path + "_flat.csv")
    elif df_format == "pickle":
        df_table.to_pickle(df_file_path + "_table.pkl")
        df_flat.to_pickle(df_file_path + "_flat.pkl")


def create_dataframe_for_experiment(
    config_dict: Dict[str, Any],
    metric_list: List[str],
    sections: List[str],
    df_format: str = "pickle",
):
    """
    Given a list of sections, merges the metric results from each DataFrame in a single DataFrame, saving it as **experiment**
    section.

    Parameters
    ----------
     config_dict : Dict[str, Any]
        Configuration dictionary, including experiment settings.
    metric_list : List[str]
        List of metrics.
    sections : List[str]
        List of section names to merge. Values accepted: ``validation``, ``testing``.
    df_format : str
        File format to save the Pandas DataFrame, can be Excel, CSV or Pickle. Defaults to Pickle.
    """
    pd_flat_metric_list_summary = []
    pd_metric_list_summary = []

    for metric_name in metric_list:
        df_list = []
        df_flat_list = []
        for section in sections:
            df_path = str(
                Path(config_dict["results_folder"]).joinpath(METRICS_FOLDER_NAME, section, metric_name, "{}".format(metric_name))
            )
            if df_format == "csv":
                df_list.append(read_dataframe(df_path + "_table.csv"))
                df_flat_list.append(read_dataframe(df_path + "_flat.csv"))
            elif df_format == "pickle":
                df_list.append(read_dataframe(df_path + "_table.pkl"))
                df_flat_list.append(read_dataframe(df_path + "_flat.pkl"))
            elif df_format == "excel":
                df_list.append(read_dataframe(df_path + ".xlsx", sheet_name="Table"))
                df_flat_list.append(read_dataframe(df_path + ".xlsx", sheet_name="Flat"))

        df = pd.concat(df_list, ignore_index=True)
        df_flat = pd.concat(df_flat_list, ignore_index=True)

        df_flat_summary = df_flat.rename(columns={metric_name: "Metric_Score"})
        df_flat_summary["Metric"] = metric_name
        pd_flat_metric_list_summary.append(df_flat_summary)
        df_summary = df.copy(deep=True)
        df_summary["Metric"] = metric_name
        pd_metric_list_summary.append(df_summary)

        Path(config_dict["results_folder"]).joinpath(METRICS_FOLDER_NAME, "experiment", metric_name).mkdir(
            exist_ok=True, parents=True
        )

        subject_list = set(df_flat["Subject"].tolist())
        subject_id = {subject: str(index) for index, subject in enumerate(subject_list)}
        with open(
            Path(config_dict["results_folder"]).joinpath(METRICS_FOLDER_NAME, "experiment", metric_name, "subject_id.json"), "w"
        ) as fp:
            json.dump(subject_id, fp)

        subject_table = get_subject_table(
            str(Path(config_dict["results_folder"]).joinpath(METRICS_FOLDER_NAME, "experiment", metric_name, "subject_id.json"))
        )
        subject_table.write_html(
            str(Path(config_dict["results_folder"]).joinpath(METRICS_FOLDER_NAME, "experiment", metric_name, "subject_id.html"))
        )

        df_file_path = str(
            Path(config_dict["results_folder"]).joinpath(METRICS_FOLDER_NAME, "experiment", metric_name, "{}".format(metric_name))
        )

        if df_format == "excel":
            writer = pd.ExcelWriter(df_file_path + ".xlsx", engine="xlsxwriter")
            df.to_excel(writer, sheet_name="Table", index=False)
            df_flat.to_excel(writer, sheet_name="Flat", index=False)
            writer.save()
        elif df_format == "csv":
            df.to_csv(df_file_path + "_table.csv")
            df_flat.to_csv(df_file_path + "_flat.csv")
        elif df_format == "pickle":
            df.to_pickle(df_file_path + "_table.pkl")
            df_flat.to_pickle(df_file_path + "_flat.pkl")

    df_flat = pd.concat(pd_flat_metric_list_summary, ignore_index=True)
    df_table = pd.concat(pd_metric_list_summary, ignore_index=True)
    Path(config_dict["results_folder"]).joinpath(METRICS_FOLDER_NAME).mkdir(exist_ok=True, parents=True)

    subject_list = set(df_flat["Subject"].tolist())
    subject_id = {subject: str(index) for index, subject in enumerate(subject_list)}
    with open(Path(config_dict["results_folder"]).joinpath(METRICS_FOLDER_NAME, "subject_id.json"), "w") as fp:
        json.dump(subject_id, fp)

    subject_table = get_subject_table(str(Path(config_dict["results_folder"]).joinpath(METRICS_FOLDER_NAME, "subject_id.json")))
    subject_table.write_html(str(Path(config_dict["results_folder"]).joinpath(METRICS_FOLDER_NAME, "subject_id.html")))

    df_file_path = str(Path(config_dict["results_folder"]).joinpath(METRICS_FOLDER_NAME, config_dict["Experiment Name"]))

    if df_format == "excel":
        writer = pd.ExcelWriter(df_file_path + ".xlsx", engine="xlsxwriter")
        df_table.to_excel(writer, sheet_name="Table", index=False)
        df_flat.to_excel(writer, sheet_name="Flat", index=False)
        writer.save()
    elif df_format == "csv":
        df_table.to_csv(df_file_path + "_table.csv")
        df_flat.to_csv(df_file_path + "_flat.csv")
    elif df_format == "pickle":
        df_table.to_pickle(df_file_path + "_table.pkl")
        df_flat.to_pickle(df_file_path + "_flat.pkl")


def save_dataframes(
    config_dict: Dict[str, Any],
    metric: str,
    section: str,
    result_suffix: str,
    df_format: str = "pickle",
    subject_phase_dict: Dict[str, str] = None,
):
    """
    Given a metric and a section, saves the corresponding DataFrames.

    Parameters
    ----------
    config_dict : Dict[str, Any]
        Configuration dictionary, including experiment settings.
    metric : str
        Metric name:
    section : str
        Section name. Values accepted:  ``validation``, ``testing``.
    result_suffix : str
        String used to retrieve the JSON result summaries, from where to extract metric scores.
    df_format : str
        File format to save the Pandas DataFrame, can be Excel, CSV or Pickle. Defaults to Pickle.
    subject_phase_dict : Dict[str, str]
        Optional Dictionary including breathing Phase for each Subject ID.
    """
    try:
        summary_filepath = get_results_summary_filepath(config_dict, section, result_suffix)
    except ValueError:
        return
    base_metrics = read_metric_list(summary_filepath, config_dict)
    save_metrics(config_dict, metric, base_metrics, section, result_suffix, df_format, subject_phase_dict)


def get_saved_dataframes(
    config_dict: Dict[str, Any],
    metrics: List[str],
    sections: List[str],
    df_format: str = "pickle",
) -> Dict[str, str]:
    """
    For a specified experiments, returns a map including all the saved DataFrames file paths.

    Parameters
    ----------
    config_dict : Dict[str, Any]
        Configuration dictionary, including experiment settings.
    metrics : List[str]
        List of metrics
    sections : List[str]
        List of section names to retrieve DataFrames. Accepted values are: [``"project"``, ``"experiment"``,
        ``"validation"``, ``"testing"``].
    df_format : str
        File format to save the Pandas DataFrame, can be Excel, CSV or Pickle. Defaults to Pickle.
    Returns
    -------
    Dict[str, str]
        Map containing all the saved metric DataFrames with the file paths.
    """
    file_extension = ""
    if df_format == "pickle":
        file_extension = "pkl"
    elif df_format == "csv":
        file_extension = "csv"

    pd_gui = {}

    if "project" in sections:
        df_path = Path(config_dict["results_folder"]).joinpath(
            METRICS_FOLDER_NAME, "{}_table.{}".format(config_dict["ProjectName"], file_extension)
        )
        df_flat_path = Path(config_dict["results_folder"]).joinpath(
            METRICS_FOLDER_NAME, "{}_flat.{}".format(config_dict["ProjectName"], file_extension)
        )

        if df_format == "excel":
            df_path = Path(config_dict["results_folder"]).joinpath(
                METRICS_FOLDER_NAME, "{}.xlsx".format(config_dict["ProjectName"])
            )
            df_flat_path = Path(config_dict["results_folder"]).joinpath(
                METRICS_FOLDER_NAME, "{}.xlsx".format(config_dict["ProjectName"])
            )
        if df_path.is_file():
            pd_gui["{}".format(config_dict["ProjectName"])] = str(df_path)
        if df_flat_path.is_file():
            pd_gui["{}_flat".format(config_dict["ProjectName"])] = str(df_flat_path)
        return pd_gui

    for metric in metrics:
        for section in sections:
            df_path = Path(config_dict["results_folder"]).joinpath(
                METRICS_FOLDER_NAME, section, metric, "{}_table.{}".format(metric, file_extension)
            )
            df_flat_path = Path(config_dict["results_folder"]).joinpath(
                METRICS_FOLDER_NAME, section, metric, "{}_flat.{}".format(metric, file_extension)
            )
            if df_format == "excel":
                df_path = Path(config_dict["results_folder"]).joinpath(
                    METRICS_FOLDER_NAME, section, metric, "{}.xlsx".format(metric)
                )
                df_flat_path = Path(config_dict["results_folder"]).joinpath(
                    METRICS_FOLDER_NAME, section, metric, "{}.xlsx".format(metric)
                )

            if df_path.is_file():
                pd_gui[metric + "_" + section] = str(df_path)
            if df_flat_path.is_file():
                pd_gui[metric + "_flat" + "_" + section] = str(df_flat_path)
    return pd_gui


def read_dataframe(df_path: Union[str, PathLike], sheet_name: str = None) -> DataFrame:
    """
    Reads and returns a Pandas DataFrame, automatically detecting the file format. If the format is Excel, *sheet_name*
    is specified to select which Excel sheet to load.

    Parameters
    ----------
    df_path : Union[str, PathLike]
        File path to load as Pandas DataFrame
    sheet_name : str
        Optional sheet name to choose when loading DataFrame from Excel file
    Returns
    -------
    DataFrame
        Pandas DataFrame stored in the file
    """
    file_extension = Path(df_path).suffix
    if file_extension == ".pkl":
        return pd.read_pickle(df_path)
    elif file_extension == ".csv":
        return pd.read_csv(df_path)
    elif file_extension == ".xlsx":
        return pd.read_excel(df_path, sheet_name=sheet_name)


def create_dataframes(
    config_dict: Dict[str, Any],
    metrics: List[str],
    sections: List[str],
    result_suffix: str = "",
    df_format: str = "pickle",
    subject_phase_dict: Dict[str, str] = None,
):
    """
    Creates set of Pandas dataframes, given a metric list and the prediction suffix, used to retrieve the corresponding
    JSON summaries. For each metric, 3 set dataframes are created: for validation results, for testing results and a
    global one, including both.
    The Pandas Dataframes are saved in */PATH/TO/RESULTS/METRICS_FOLDER_NAME/SECTION/METRIC*, with section = [``"experiment"``,
     ``"validation"``, ``"testing"``].

    Parameters
    ----------
    config_dict : Dict[str, Any]
        Configuration dictionary, including experiment settings.
    metrics : List[str]
        List of metrics to create the Pandas dataframes.
    sections : List[str]
        List of section names to create DataFrames. Accepted values are: [``"experiment"``, ``"validation"``, ``"testing"``].
    result_suffix: str
        String used to retrieve the JSON result summaries, from where to extract metrics scores.
    df_format : str
        File format to save the Pandas DataFrame, can be Excel, CSV or Pickle. Defaults to Pickle.
    subject_phase_dict : Dict[str, str]
        Optional Dictionary including breathing Phase for each Subject ID.
    """
    for metric in metrics:
        for section in sections:
            save_dataframes(config_dict, metric, section, result_suffix, df_format, subject_phase_dict)
    if "experiment" in sections:
        sections_to_combine = sections.copy()
        sections_to_combine.remove("experiment")
        create_dataframe_for_experiment(config_dict, metrics, sections_to_combine, df_format)
