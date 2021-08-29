#!/usr/bin/env python

import argparse
import json
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from textwrap import dedent

import pandas as pd
import plotly.io as pio
import visdom
from pandasgui import show

from Hive.evaluation.io_metric_results import (
    create_dataframes,
    get_saved_dataframes,
    METRICS_FOLDER_NAME,
)
from Hive.evaluation.plotly_plots import create_plots, save_plots, PLOTS, BAR_AGGREGATORS
from Hive.evaluation.vis import create_log_at
from Hive.utils.log_utils import get_logger, add_verbosity_options_to_argparser, log_lvl_from_verbosity_args

pio.renderers.default = "browser"

DESC = dedent(
    """
    Generates metric result tables as ``Pandas Dataframe``, saving them as ``Pickle`` files. The files are stored in
    */path/to/results_folder/METRICS_FOLDER/SECTION/METRIC_NAME*, with SECTION indicating ``Experiment``, ``Validation`` 
    or ``Testing`` metrics.
    The created **Pandas Dataframes** can optionally be inspected with ```PandasGui``.
    For each given metric, a table with the metric score for each label class is created, including a flat version used for ``Plotly`` visualization.
    Optionally, a Pandas Dataframe including the average and the standard deviation scores for each label class is saved.
    The selected metrics can be included in the set of basic metrics, or they can be an advanced combination of the basic ones.
    The basic metrics are: [ ``"Dice"``, ``"Accuracy"``, ``"Jaccard"``, ``"Recall"``, ``"Precision"``, ``"False Positive Rate"``,
    ``"False Omission Rate"``, ``"Hausdorff Distance"``, ``"Hausdorff Distance 95"``, ``"Avg. Surface Distance"``,
    ``"Avg. Symmetric Surface Distance"``, ``"True Positives"``, ``"False Positives"``, ``"True negatives"``, ``"False Negatives"``,
    ``"Total Positives Test"``, ``"Total Positives Reference"``]

    Advanced metric combination example:
        .. math::
            Specificity = 1 - False Positive Rate     
    Optionally, a histogram and a boxplot distributions are rendered, representing the statistics for each label class.
    The plots can be either visualized in a browser window or saved as PNG files or HTML files. If a Visdom server is running, it is
    also possible to upload the metric plots and the statistics tables on the server.
    """  # noqa: E501 W291
)
EPILOG = dedent(
    """
    Example call:
    ::
        {filename} --config-file /path/to/config_file.json --section testing 
        {filename} --config-file /path/to/config_file.json --section testing --metrics Dice Accuracy Hausdorff Distance
        {filename} --config-file /path/to/config_file.json --section testing --visualize-only True --sections testing validation
        {filename} --config-file /path/to/config_file.json --section testing --visualize-only True  --save-png True
        {filename} --config-file /path/to/config_file.json --section testing --display-in-browser True  --save-png True
        {filename} --config-file /path/to/config_file.json --section testing --upload-visdom-server True  --save-png True
    """.format(  # noqa: E501 W291
        filename=Path(__file__).name
    )
)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_arg_parser():
    pars = ArgumentParser(description=DESC, epilog=EPILOG, formatter_class=RawTextHelpFormatter)

    pars.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="File path for the configuration dictionary, used to retrieve experiment settings ",
    )

    pars.add_argument(
        "--metrics",
        type=str,
        required=False,
        nargs="+",
        help="Sequence of metrics to be computed. If specified, the metrics listed in the configuration file are overridden",
    )

    pars.add_argument(
        "--sections",
        type=str,
        required=False,
        nargs="+",
        help="Sequence of sections to compute the metrics on. Values can be: [ testing, validation, experiment ].",
    )

    pars.add_argument(
        "--visualize-only",
        type=str2bool,
        required=False,
        default=False,
        help="Visualize results only, without creating the corresponding **Pandas Dataframes**",
    )

    pars.add_argument(
        "--save-png",
        type=str2bool,
        required=False,
        default=False,
        help="Specify to save the metric **Plotly** plots as PNGs",
    )

    pars.add_argument(
        "--save-json",
        type=str2bool,
        required=False,
        default=False,
        help="Specify to save the metric **Plotly** plots as JSON",
    )

    pars.add_argument(
        "--save-html",
        type=str2bool,
        required=False,
        default=False,
        help="Specify to save the metric **Plotly** plots as HTMLs",
    )

    pars.add_argument(
        "--show-in-browser",
        type=str2bool,
        required=False,
        default=False,
        help="Specify to display the metric **Plotly** plots in the default browser",
    )

    pars.add_argument(
        "--show-pandas-gui",
        type=str2bool,
        required=False,
        default=False,
        help="Specify to inspect the Pandas Dataframes using **PandasGUI**",
    )

    pars.add_argument(
        "--upload-visdom-server",
        type=str2bool,
        required=False,
        default=False,
        help="Specify to upload the metric **Plotly** plots in the running **Visdom** server",
    )

    pars.add_argument(
        "--prediction-suffix",
        type=str,
        required=False,
        default="",
        help="Prediction name suffix to find the corresponding prediction files to evaluate. Defaults to ``" "`` ",
    )

    add_verbosity_options_to_argparser(pars)
    return pars


def main():
    parser = get_arg_parser()

    args = vars(parser.parse_args())

    logger = get_logger(  # NOQA: F841
        name=Path(__file__).name,
        level=log_lvl_from_verbosity_args(args),
    )

    prediction_suffix = args["prediction_suffix"]
    if prediction_suffix != "":
        prediction_suffix = "_" + prediction_suffix

    with open(args["config_file"]) as json_file:
        config_dict = json.load(json_file)

    metrics = []

    if "Metrics_save_configs" in config_dict and "Metrics_dict" in config_dict["Metrics_save_configs"]:
        metrics_dict = config_dict["Metrics_save_configs"]["Metrics_dict"]
        if isinstance(metrics_dict, dict):
            metrics = list(metrics_dict.keys())
        elif isinstance(metrics_dict, list):
            metrics = metrics_dict

    if args["metrics"]:
        metrics = args["metrics"]

    sections = ["testing", "validation", "experiment"]
    if args["sections"]:
        sections = args["sections"]

    if args["upload_visdom_server"] is True:
        vis = visdom.Visdom()

    if args["visualize_only"] is not True:
        create_dataframes(config_dict, metrics, sections, prediction_suffix)

    df_paths = get_saved_dataframes(config_dict, metrics, sections)

    if (
            args["save_png"] is True
            or args["save_json"] is True
            or args["save_html"] is True
            or args["show_in_browser"] is True
            or args["upload_visdom_server"] is True
    ):
        plot_dict = create_plots(config_dict, df_paths, metrics, config_dict["Experiment Name"], sections)

        if args["save_png"] is True:
            save_plots(config_dict["results_folder"], plot_dict, metrics, sections, "png")

        if args["save_json"] is True:
            save_plots(config_dict["results_folder"], plot_dict, metrics, sections, "json")

        if args["show_in_browser"] is True:
            for plot in plot_dict:
                plot_dict[plot].show()

        if args["save_html"] is True:
            save_plots(config_dict["results_folder"], plot_dict, metrics, sections, "html")

        if args["upload_visdom_server"] is True:
            for metric in metrics:
                for section in sections:
                    for plot in PLOTS:
                        if plot == "bar":
                            for aggr in BAR_AGGREGATORS:
                                vis.plotlyplot(plot_dict["{}_{}_{}_{}".format(aggr, metric, section, plot)], env=metric)
                        else:
                            vis.plotlyplot(plot_dict["{}_{}_{}".format(metric, section, plot)], env=metric)
                create_log_at(
                    Path(config_dict["results_folder"]).joinpath(METRICS_FOLDER_NAME, "experiment", metric,
                                                                 metric + ".log"),
                    metric,
                )
    if args["show_pandas_gui"] is True:
        show(**{metric: pd.read_pickle(df_paths[metric]) for metric in df_paths})


if __name__ == "__main__":
    main()
