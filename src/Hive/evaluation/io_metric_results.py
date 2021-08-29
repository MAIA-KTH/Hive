import copy
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from Hive.evaluation import METRICS_FOLDER_NAME
from Hive.utils.log_utils import get_logger, DEBUG

logger = get_logger(__name__)

ADVANCED_METRICS = {
    "Specificity": {
        "Base_Metrics": ["False Positive Rate"],
        "Function": lambda x: 1 - x,
    },
    "Fowlkes–Mallows index": {"Base_Metrics": ["Precision", "Recall"],
                              "Function": lambda x, y: np.sqrt(np.multiply(x, y))},
    "Relative Volumetric Difference": {
        "Base_Metrics": ["Total Positives Reference", "Total Positives Test"],
        "Function": lambda x, y: np.divide(np.subtract(y, x), x) * 100,
    },
    "Relative Absolute Volumetric Difference": {
        "Base_Metrics": ["Total Positives Reference", "Total Positives Test"],
        "Function": lambda x, y: np.divide(np.abs(np.subtract(y, x)), x) * 100,
    },
}  # type: Dict[str, Any]


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


def read_metric_list(summary_filepath: str, config_dict: Dict[str, Any]) -> List[str]:
    """

    Parameters
    ----------
    summary_filepath : str
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

    label_dict = config_dict["label_dict"]
    label_dict.pop("0", None)
    metric_list = list(results_summary_data["results"]["all"][0][list(label_dict.keys())[0]].keys())

    return metric_list


def save_metrics(config_dict: Dict[str, Any], metric_name: str, basic_metrics: List[str], section: str,
                 result_suffix: str):
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

    """
    pd.options.display.float_format = "{:.2%}".format

    Path(config_dict["results_folder"]).joinpath(METRICS_FOLDER_NAME, section, metric_name).mkdir(exist_ok=True,
                                                                                                  parents=True)

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

    writer = pd.ExcelWriter(
        str(
            Path(config_dict["results_folder"]).joinpath(METRICS_FOLDER_NAME, section, metric_name,
                                                         "{}.xlsx".format(metric_name))
        ),
        engine="xlsxwriter",
    )

    for fold in range(n_folds):

        summary_filepath = get_results_summary_filepath(config_dict, section, result_suffix, fold)
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
                    df_temp[column_selection].loc[[metric_name]].rename(columns=column_rename,
                                                                        index={metric_name: str(subj_id)})
                )

            if column_id is not None:
                df_single_temp["ID"] = Path(df_temp[column_id][0]).name[: -len(config_dict["FileExtension"])]

            if section == "testing":
                df_single_temp["Section"] = "Testing"
            else:
                df_single_temp["Section"] = "Fold {}".format(fold)

            df_single_temp["Experiment"] = config_dict["Experiment Name"]
            df = df.append(df_single_temp)
            subj_id = subj_id + 1

    if composed_metric is not None:
        metric_name = composed_metric

    if "Save_stats" in config_dict["Metrics_save_configs"]:
        if config_dict["Metrics_save_configs"]["Save_stats"]:
            df_aggregate = pd.DataFrame(zip(df.mean(), df.std()), columns=["Mean", "SD"], index=label_dict).rename(
                index=label_dict
            )

            df_aggregate.to_pickle(
                str(
                    Path(config_dict["results_folder"]).joinpath(
                        METRICS_FOLDER_NAME, section, metric_name, "{}_stats.pkl".format(metric_name)
                    )
                )
            )

            df_aggregate.to_excel(writer, sheet_name="Stats")

    df_flat = df[[label_dict[key] for key in label_dict]].stack()
    df_flat = pd.DataFrame(df_flat)
    df_flat.reset_index(inplace=True)
    df_flat.columns = ["Subject", "Label", metric_name]

    df_flat["Section"] = section.capitalize()
    df_flat["Experiment"] = config_dict["Experiment Name"]

    df.to_pickle(
        str(
            Path(config_dict["results_folder"]).joinpath(
                METRICS_FOLDER_NAME, section, metric_name, "{}_table.pkl".format(metric_name)
            )
        )
    )

    df.to_excel(writer, sheet_name="Table", index=False)
    writer.save()
    df_flat.to_pickle(
        str(
            Path(config_dict["results_folder"]).joinpath(
                METRICS_FOLDER_NAME, section, metric_name, "{}_flat.pkl".format(metric_name)
            )
        )
    )


def create_dataframe_for_experiment(config_dict: Dict[str, Any], metric_name: str, sections: List[str]):
    """
    Given a list of sections, merges the metric results from each DataFrame in a single DataFrame, saving it as **experiment**
    section.

    Parameters
    ----------
     config_dict : Dict[str, Any]
        Configuration dictionary, including experiment settings.
    metric_name : str
        Metric name.
    sections : List[str]
        List of section names to merge.
    """
    df_list = []
    df_flat_list = []
    for section in sections:
        df_list.append(
            pd.read_pickle(
                str(
                    Path(config_dict["results_folder"]).joinpath(
                        METRICS_FOLDER_NAME, section, metric_name, "{}_table.pkl".format(metric_name)
                    )
                )
            )
        )
        df_flat_list.append(
            pd.read_pickle(
                str(
                    Path(config_dict["results_folder"]).joinpath(
                        METRICS_FOLDER_NAME, section, metric_name, "{}_flat.pkl".format(metric_name)
                    )
                )
            )
        )

    df = pd.concat(df_list, ignore_index=True)
    df_flat = pd.concat(df_flat_list, ignore_index=True)

    Path(config_dict["results_folder"]).joinpath(METRICS_FOLDER_NAME, "experiment", metric_name).mkdir(
        exist_ok=True, parents=True
    )
    df_file_path = str(
        Path(config_dict["results_folder"]).joinpath(
            METRICS_FOLDER_NAME, "experiment", metric_name, "{}_table.pkl".format(metric_name)
        )
    )
    df_flat_file_path = str(
        Path(config_dict["results_folder"]).joinpath(
            METRICS_FOLDER_NAME, "experiment", metric_name, "{}_flat.pkl".format(metric_name)
        )
    )
    writer = pd.ExcelWriter(
        str(
            Path(config_dict["results_folder"]).joinpath(
                METRICS_FOLDER_NAME, "experiment", metric_name, "{}.xlsx".format(metric_name)
            )
        ),
        engine="xlsxwriter",
    )
    df.to_excel(writer, sheet_name="Table", index=False)

    writer.save()
    df.to_pickle(df_file_path)

    df_flat.to_pickle(df_flat_file_path)


def save_dataframes(config_dict: Dict[str, Any], metric: str, section: str, result_suffix: str):
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
    """
    try:
        summary_filepath = get_results_summary_filepath(config_dict, section, result_suffix)
    except ValueError:
        return
    base_metrics = read_metric_list(summary_filepath, config_dict)
    save_metrics(config_dict, metric, base_metrics, section, result_suffix)


def get_saved_dataframes(config_dict: Dict[str, Any], metrics: List[str], sections: List[str]) -> Dict[str, str]:
    """
    For a specified experiments, returns a map including all the saved DataFrames file paths.

    Parameters
    ----------
    config_dict : Dict[str, Any]
        Configuration dictionary, including experiment settings.
    metrics : List[str]
        List of metrics
    sections : List[str]
        List of section names to retrieve DataFrames. Accepted values are: [``"experiment"``, ``"validation"``, ``"testing"``].
    Returns
    -------
    Dict[str, str]
        Map containing all the saved metric DataFrames with the file paths.
    """
    pd_gui = {}
    for metric in metrics:
        for section in sections:
            df_path = Path(config_dict["results_folder"]).joinpath(
                METRICS_FOLDER_NAME, section, metric, "{}_table.pkl".format(metric)
            )
            df_flat_path = Path(config_dict["results_folder"]).joinpath(
                METRICS_FOLDER_NAME, section, metric, "{}_flat.pkl".format(metric)
            )
            if df_path.is_file():
                pd_gui[metric + "_" + section] = str(df_path)
            if df_flat_path.is_file():
                pd_gui[metric + "_flat" + "_" + section] = str(df_flat_path)
    return pd_gui


def create_dataframes(config_dict: Dict[str, Any], metrics: List[str], sections: List[str], result_suffix: str = ""):
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

    """
    for metric in metrics:
        for section in sections:
            save_dataframes(config_dict, metric, section, result_suffix)
        if "experiment" in sections:
            sections_to_combine = sections.copy()
            sections_to_combine.remove("experiment")
            create_dataframe_for_experiment(config_dict, metric, sections_to_combine)
