#!/usr/bin/env python

from textwrap import dedent
import json
from pathlib import Path
import visdom
import pandas as pd
from pandasgui import show
import plotly.io as pio
import argparse
from argparse import ArgumentParser, RawTextHelpFormatter
from k8s_DP.utils.log_utils import get_logger, add_verbosity_options_to_argparser, log_lvl_from_verbosity_args
from k8s_DP.evaluation.io_metric_results import (
    get_results_summary_filepath,
    read_metric_list,
    save_metrics,
    SECTIONS,
    get_plotly_histo,
    get_metric_stats_as_html_table,
    get_plotly_boxplot,
    DEFAULT_BAR_CONFIGS,
    DEFAULT_METRIC_UNITS,
)

pio.renderers.default = "browser"

DESC = dedent(
    """
    Generates metric result tables as ``Pandas Dataframe``, saving them as ``Pickle`` files. The files are stored in
    */path/to/results_folder/metrics_DF/SECTION/METRIC_NAME*, with SECTION indicating ``Validation`` or ``Testing`` metrics.
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
        {filename} --config-file /path/to/config_file.json --section testing --visualize-only True
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
        "--section",
        type=str,
        required=True,
        help="Experiment section to evaluate the metrics. Choices: {} ".format(SECTIONS),
    )

    pars.add_argument(
        "--metrics",
        type=str,
        required=False,
        nargs="+",
        help="Sequence of metrics to be computed. If specified, the metrics listed in the configuration file are overridden",
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

    add_verbosity_options_to_argparser(pars)
    return pars


def main():
    parser = get_arg_parser()

    args = vars(parser.parse_args())

    logger = get_logger(  # NOQA: F841
        name=Path(__file__).name,
        level=log_lvl_from_verbosity_args(args),
    )

    if args["section"] not in SECTIONS:
        raise ValueError("Invalid section. Expected one of: %s" % SECTIONS)

    with open(args["config_file"]) as json_file:
        config_dict = json.load(json_file)

    if args["visualize_only"] is not True:
        summary_filepath = get_results_summary_filepath(config_dict, args["section"])
        base_metrics = read_metric_list(summary_filepath, config_dict)

    metrics = []
    metrics_dict = None
    if "Metrics_save_configs" in config_dict and "Metrics_dict" in config_dict["Metrics_save_configs"]:
        metrics_dict = config_dict["Metrics_save_configs"]["Metrics_dict"]
        if isinstance(metrics_dict, dict):
            metrics = list(metrics_dict.keys())
        elif isinstance(metrics_dict, list):
            metrics = metrics_dict

    if args["metrics"]:
        metrics = args["metrics"]

    if args["upload_visdom_server"] is True:
        vis = visdom.Visdom()

    label_dict = config_dict["label_dict"]
    label_dict.pop("0", None)

    pd_gui = {}
    for metric in metrics:

        if args["visualize_only"] is not True:
            save_metrics(config_dict, metric, base_metrics, args["section"])

        df_path = Path(config_dict["results_folder"]).joinpath(
            "metrics_DF", args["section"], metric, "{}_table.pkl".format(metric)
        )
        df_flat_path = Path(config_dict["results_folder"]).joinpath(
            "metrics_DF", args["section"], metric, "{}_flat.pkl".format(metric)
        )
        if df_path.is_file():
            pd_gui[metric] = df_path
            pd_gui[metric + "_flat"] = df_flat_path

        if (
            args["save_png"] is True
            or args["save_html"] is True
            or args["show_in_browser"] is True
            or args["upload_visdom_server"] is True
        ):

            if df_flat_path.is_file():
                df_flat = pd.read_pickle(str(df_flat_path))
                if metrics_dict is not None and isinstance(metrics_dict, dict) and "m_unit" in metrics_dict[metric]:
                    measurement_unit = metrics_dict[metric]["m_unit"]
                else:
                    if metric in DEFAULT_METRIC_UNITS:
                        measurement_unit = DEFAULT_METRIC_UNITS[metric]
                    else:
                        measurement_unit = ""

                fig_histo = get_plotly_histo(df_flat, metric, measurement_unit, args["section"])
                fig_boxplot = get_plotly_boxplot(df_flat, metric, measurement_unit, args["section"])
                if args["save_png"] is True:
                    fig_histo.write_image(
                        str(
                            Path(config_dict["results_folder"]).joinpath(
                                "metrics_DF", args["section"], metric, "{}_histo.png".format(metric)
                            )
                        )
                    )
                    fig_boxplot.write_image(
                        str(
                            Path(config_dict["results_folder"]).joinpath(
                                "metrics_DF", args["section"], metric, "{}_boxplot.png".format(metric)
                            )
                        )
                    )
                if args["show_in_browser"] is True:
                    fig_histo.show()
                    fig_boxplot.show()

                if args["save_html"] is True:
                    fig_histo.write_html(
                        str(
                            Path(config_dict["results_folder"]).joinpath(
                                "metrics_DF", args["section"], metric, "{}_histo.html".format(metric)
                            )
                        )
                    )
                    fig_boxplot.write_html(
                        str(
                            Path(config_dict["results_folder"]).joinpath(
                                "metrics_DF", args["section"], metric, "{}_boxplot.html".format(metric)
                            )
                        )
                    )

                    if df_path.is_file():
                        df = pd.read_pickle(str(df_path))
                        if metric in DEFAULT_BAR_CONFIGS:
                            bar_configs = DEFAULT_BAR_CONFIGS[metric]
                        else:
                            bar_configs = None

                        if "Metrics_save_configs" in config_dict and "Metrics_dict" in config_dict["Metrics_save_configs"]:
                            if (
                                isinstance(config_dict["Metrics_save_configs"]["Metrics_dict"], dict)
                                and metric in config_dict["Metrics_save_configs"]["Metrics_dict"]
                            ):
                                if "bar_config" in config_dict["Metrics_save_configs"]["Metrics_dict"][metric]:
                                    bar_configs = config_dict["Metrics_save_configs"]["Metrics_dict"][metric]["bar_config"]

                        metric_stats_html = get_metric_stats_as_html_table(df, label_dict, metric, args["section"], bar_configs)

                        file = open(
                            Path(config_dict["results_folder"]).joinpath(
                                "metrics_DF", args["section"], metric, "{}_stats.html".format(metric)
                            ),
                            "w",
                        )

                        file.write(metric_stats_html)

                        file.close()

                if args["upload_visdom_server"] is True:
                    vis.plotlyplot(fig_histo, env=metric)
                    vis.plotlyplot(fig_boxplot, env=metric)
                    df_path = Path(config_dict["results_folder"]).joinpath(
                        "metrics_DF", args["section"], "{}_table.pkl".format(metric)
                    )
                    if df_path.is_file():
                        df = pd.read_pickle(str(df_path))

                        if metric in DEFAULT_BAR_CONFIGS:
                            bar_configs = DEFAULT_BAR_CONFIGS[metric]
                        else:
                            bar_configs = None
                        if "Metrics_save_configs" in config_dict and "Metrics_dict" in config_dict["Metrics_save_configs"]:
                            if (
                                isinstance(config_dict["Metrics_save_configs"]["Metrics_dict"], dict)
                                and metric in config_dict["Metrics_save_configs"]["Metrics_dict"]
                            ):
                                if "bar_config" in config_dict["Metrics_save_configs"]["Metrics_dict"][metric]:
                                    bar_configs = config_dict["Metrics_save_configs"]["Metrics_dict"][metric]["bar_config"]

                        metric_stats_html = get_metric_stats_as_html_table(df, label_dict, metric, args["section"], bar_configs)
                        vis.text(metric_stats_html, env=metric)

                    else:
                        logger.info("{} does not exist".format(str(df_path)))
            else:
                logger.info("{} does not exist".format(str(df_flat_path)))

    if args["show_pandas_gui"] is True:
        show(**{metric: pd.read_pickle(pd_gui[metric]) for metric in pd_gui})


if __name__ == "__main__":
    main()
