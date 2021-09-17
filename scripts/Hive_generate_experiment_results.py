#!/usr/bin/env python

import json
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from textwrap import dedent

import plotly.io as pio
import visdom

from Hive.evaluation.io_metric_results import (
    create_dataframes,
    get_saved_dataframes,
    METRICS_FOLDER_NAME,
    read_dataframe,
)
from Hive.evaluation.plotly_plots import create_plots, save_plots, PLOTS, BAR_AGGREGATORS
from Hive.evaluation.vis import create_log_at
from Hive.utils.log_utils import get_logger, add_verbosity_options_to_argparser, log_lvl_from_verbosity_args, str2bool

pio.renderers.default = "browser"

DESC = dedent(
    """
    Generates metric result tables as ``Pandas Dataframe``, saving them as ``Pickle``, ``Excel`` or ``CSV`` files.
    The files are stored in */path/to/results_folder/METRICS_FOLDER/SECTION/METRIC_NAME*, with SECTION indicating
    ``Experiment``, ``Validation`` or ``Testing`` metrics.
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
            Specificity = 1 - False\ Positive\ Rate     
    Optionally, a histogram, a bar plot and a boxplot distributions are rendered, representing the statistics for each label class.
    The plots can be either visualized in a browser window or saved as PNG files, JSON or HTML files. If a Visdom server is running, it is
    also possible to upload the metric plots and the statistics tables on the server.
    """  # noqa: E501 W291 W605
)
EPILOG = dedent(
    """
    Example call:
    ::
        {filename} --config-file /path/to/config_file.json
        {filename} --config-file /path/to/config_file.json --metrics Dice Accuracy Hausdorff Distance
        {filename} --config-file /path/to/config_file.json --visualize-only True --sections testing validation
        {filename} --config-file /path/to/config_file.json --visualize-only True  --save-png True
        {filename} --config-file /path/to/config_file.json --display-in-browser True  --save-png True
        {filename} --config-file /path/to/config_file.json --display-in-browser True  --df-format pickle
        {filename} --config-file /path/to/config_file.json --upload-visdom-server True  --save-png True
    """.format(  # noqa: E501 W291
        filename=Path(__file__).name
    )
)


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
        help="Sequence of sections to compute the metrics on. Values can be: [ ``testing``, ``validation``, ``experiment`` ].",
    )

    pars.add_argument(
        "--df-format",
        type=str,
        required=False,
        default="pickle",
        help="File format to use to save Pandas DataFrame. Values can be: [ ``excel``, ``csv``, ``pickle`` ].",
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

    pars.add_argument(
        "--phase-json-file",
        type=str,
        required=False,
        default=None,
        help="Breathing phase JSON file, including breathing phase for each subject ID.",
    )

    pars.add_argument(
        "--plot-phase",
        type=str2bool,
        required=False,
        default=False,
        help="Specify to create plots including breathing phase information.",
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

    file_format = args["df_format"]

    if args["upload_visdom_server"] is True:
        vis = visdom.Visdom()

    phase_dict = None
    if args["phase_json_file"] is not None:
        with open(args["phase_json_file"]) as json_file:
            phase_dict = json.load(json_file)

    if args["visualize_only"] is not True:
        create_dataframes(config_dict, metrics, sections, prediction_suffix, file_format, phase_dict)

    df_paths = get_saved_dataframes(config_dict, metrics, sections, file_format)
    if (
        args["save_png"] is True
        or args["save_json"] is True
        or args["save_html"] is True
        or args["show_in_browser"] is True
        or args["upload_visdom_server"] is True
    ):
        plot_dict = create_plots(config_dict, df_paths, metrics, config_dict["Experiment Name"], sections, args["plot_phase"])

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
                    Path(config_dict["results_folder"]).joinpath(METRICS_FOLDER_NAME, "experiment", metric, metric + ".log"),
                    metric,
                )
    if args["show_pandas_gui"] is True:
        df_dict = {}
        for metric in df_paths:
            if file_format == "excel":
                if "flat" in metric:
                    df_dict[metric] = read_dataframe(df_paths[metric], sheet_name="Flat")
                else:
                    df_dict[metric] = read_dataframe(df_paths[metric], sheet_name="Table")
            else:
                df_dict[metric] = read_dataframe(df_paths[metric])
        from pandasgui import show

        show(**df_dict)


if __name__ == "__main__":
    main()
