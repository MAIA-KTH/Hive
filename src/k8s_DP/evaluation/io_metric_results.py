import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import numpy as np
import copy

from k8s_DP.utils.file_utils import subfiles
from k8s_DP.utils.log_utils import get_logger, DEBUG

SECTIONS = ["testing", "validation"]

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
}

DEFAULT_BAR_CONFIGS = {
    "Dice": {"thresholds": [0.9, 0.95], "colors": ["green", "orange", "red"], "min_value": 0.0, "max_value": 1.0},
    "Hausdorff Distance": {"thresholds": [40, 80], "colors": ["red", "orange", "green"], "min_value": 0.0},
}

DEFAULT_METRIC_UNITS = {"Dice": "", "Hausdorff Distance": "[ mm ]"}


def find_file_from_pattern(folder, pattern, file_extension):
    files = subfiles(folder, prefix=pattern[: -len(file_extension)], suffix=file_extension)
    if len(files) > 0:
        return files[0]
    else:
        return None


def get_results_summary_filepath(config_dict, section, fold=0):
    if section not in SECTIONS:
        raise ValueError("Invalid section. Expected one of: %s" % SECTIONS)

    full_task_name = "Task" + config_dict["Task_ID"] + "_" + config_dict["Task_Name"]
    fold_folder = "fold_" + str(fold)
    if section == "validation":
        parent_folder = str(Path(fold_folder).joinpath("validation_raw_postprocessed"))
    else:
        parent_folder = "predictionsTs"
    summary_filepath = str(
        Path(config_dict["results_folder"]).joinpath(
            "nnUNet",
            config_dict["TRAINING_CONFIGURATION"],
            full_task_name,
            config_dict["TRAINER_CLASS_NAME"] + "__" + config_dict["TRAINER_PLAN"],
            parent_folder,
            "summary.json",
        )
    )

    return summary_filepath


def read_metric_list(summary_filepath, config_dict):
    with open(summary_filepath) as json_file:
        results_summary_data = json.load(json_file)

    label_dict = config_dict["label_dict"]
    label_dict.pop("0", None)
    metric_list = list(results_summary_data["results"]["all"][0][list(label_dict.keys())[0]].keys())

    return metric_list


def save_metrics(config_dict, metric_name, basic_metrics, section):
    pd.options.display.float_format = "{:.2%}".format

    if section not in SECTIONS:
        raise ValueError("Invalid section. Expected one of: %s" % SECTIONS)

    Path(config_dict["results_folder"]).joinpath("metrics_DF", section, metric_name).mkdir(exist_ok=True, parents=True)

    additional_columns = {}
    add_volume_column = False

    if "Metrics_save_configs" in config_dict:
        if "Additional_DF_columns" in config_dict["Metrics_save_configs"]:
            additional_columns = config_dict["Metrics_save_configs"]["Additional_DF_columns"]
        if (
            "Add_Volume_Column" in config_dict["Metrics_save_configs"]
            and "Search_file_pattern_from" in config_dict["Metrics_save_configs"]
        ):
            add_volume_column = config_dict["Metrics_save_configs"]["Add_Volume_Column"]
            volume_reference_column = config_dict["Metrics_save_configs"]["Search_file_pattern_from"]

    if section == "testing":
        n_folds = 1
        base_folder = "imagesTs"
    else:
        n_folds = config_dict["n_folds"]
        base_folder = "imagesTr"

    full_task_name = "Task" + config_dict["Task_ID"] + "_" + config_dict["Task_Name"]
    images_folder_path = str(Path(config_dict["base_folder"]).joinpath("nnUNet_raw_data", full_task_name, base_folder))

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

    for fold in range(n_folds):

        summary_filepath = get_results_summary_filepath(config_dict, section, fold)
        with open(summary_filepath) as json_file:
            summary_results_data = json.load(json_file)

        for i in [x for x in range(len(summary_results_data["results"]["all"]))]:
            df_temp = pd.DataFrame(summary_results_data["results"]["all"][i])
            column_selection = list(label_dict.keys())
            column_rename = copy.deepcopy(label_dict)
            for additional_column in list(additional_columns.keys()):
                column_selection.append(additional_column)
                column_rename[additional_column] = additional_columns[additional_column]

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

            if add_volume_column and Path(images_folder_path).is_dir():
                volume_file = find_file_from_pattern(
                    images_folder_path, Path(df_single_temp[volume_reference_column][0]).name, config_dict["FileExtension"]
                )

                if Path(volume_file).is_file():
                    df_single_temp["Volume File"] = volume_file
                else:
                    df_single_temp["Volume File"] = ""

            if section == "testing":
                df_single_temp["Section"] = "Testing"
            else:
                df_single_temp["Section"] = "Fold {}".format(fold)
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
                        "metrics_DF", section, metric_name, "{}_stats.pkl".format(metric_name)
                    )
                )
            )

    df_flat = df[[label_dict[key] for key in label_dict]].stack()
    df_flat = pd.DataFrame(df_flat)
    df_flat.reset_index(inplace=True)
    df_flat.columns = ["Subject", "Label", metric_name]

    df.to_pickle(
        str(Path(config_dict["results_folder"]).joinpath("metrics_DF", section, metric_name, "{}_table.pkl".format(metric_name)))
    )
    df_flat.to_pickle(
        str(Path(config_dict["results_folder"]).joinpath("metrics_DF", section, metric_name, "{}_flat.pkl".format(metric_name)))
    )


def get_plotly_histo(df_flat, metric_name, metric_measurement_unit, section):
    fig_histo = px.histogram(
        df_flat,
        x=metric_name,
        color="Label",
        labels={
            metric_name: metric_name + " " + metric_measurement_unit,
        },
        title="{} Set, {}".format(section.capitalize(), metric_name),
    )

    return fig_histo


def get_plotly_boxplot(df_flat, metric_name, metric_measurement_unit, section):
    fig_boxplot = px.box(
        df_flat,
        x="Label",
        y=metric_name,
        color="Label",
        labels={
            metric_name: metric_name + " " + metric_measurement_unit,
        },
        title="{} Set, {}".format(section.capitalize(), metric_name),
    )

    return fig_boxplot


def get_metric_stats_as_html_table(pandas_df, label_dict, metric_name, section, bar_configs=None):
    df_stats = pd.DataFrame(zip(pandas_df.mean(), pandas_df.std()), columns=["Mean", "SD"], index=label_dict).rename(
        index=label_dict
    )

    df_styler = (
        df_stats.style.set_table_attributes("style='display:inline'")
        .set_caption("{} Set,  {}".format(section.capitalize(), metric_name))
        .format({"Mean": "{:.3}", "SD": "{:.3}"})
        .set_table_styles(
            [{"selector": "th", "props": [("font-size", "50px")]}, {"selector": "tr", "props": [("font-size", "40px")]}]
        )
    )

    if bar_configs is not None:

        lower_threshold = bar_configs["thresholds"][0]
        upper_threshold = bar_configs["thresholds"][1]

        colors = ["green", "orange", "red"]
        if "colors" in bar_configs:
            colors = bar_configs["colors"]

        if "max_value" in bar_configs:
            max_val = bar_configs["max_value"]
        else:
            max_val = df_stats["Mean"].max()

        if "min_value" in bar_configs:
            min_val = bar_configs["min_value"]
        else:
            min_val = df_stats["Mean"].min()

        i_high = pd.IndexSlice[df_stats.loc[(df_stats["Mean"] >= upper_threshold)].index, "Mean"]
        i_mid = pd.IndexSlice[
            df_stats.loc[((df_stats["Mean"] < upper_threshold) & (df_stats["Mean"] >= lower_threshold))].index, "Mean"
        ]
        i_low = pd.IndexSlice[df_stats.loc[(df_stats["Mean"] < lower_threshold)].index, "Mean"]

        df_styler.bar(subset=i_high, color=colors[0], align="left", vmin=min_val, vmax=max_val).bar(
            subset=i_mid, color=colors[1], align="left", vmin=min_val, vmax=max_val
        ).bar(subset=i_low, color=colors[2], align="left", vmin=min_val, vmax=max_val)
    df_stats_html = df_styler.render()

    return df_stats_html
