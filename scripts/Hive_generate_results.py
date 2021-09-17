#!/usr/bin/env python

import json
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from textwrap import dedent

import visdom

from Hive.evaluation import METRICS_FOLDER_NAME
from Hive.evaluation.io_metric_results import create_dataframe_for_project, get_saved_dataframes
from Hive.evaluation.plotly_plots import create_plots_for_project
from Hive.evaluation.plotly_plots import save_plots, PLOTS, BAR_AGGREGATORS
from Hive.evaluation.vis import create_log_at
from Hive.utils.log_utils import str2bool, add_verbosity_options_to_argparser, get_logger, log_lvl_from_verbosity_args

DESC = dedent(
    """
    Generates metric result tables as ``Pandas Dataframe``, grouping all the experiment results on the specific project,
    saving them as ``Pickle``, ``Excel`` or ``CSV`` files.
    The files are stored in */path/to/project_results_folder/METRICS_FOLDER/METRIC_NAME*.
    A project summary is also saved as JSON file.
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
        {filename} --config-files /path/to/config_file_A.json /path/to/config_file_B.json --results-folder /path/to/results_folder --project-name Project_A 
        {filename} --config-files /path/to/config_file.json /path/to/config_file_B.json /path/to/config_file_c.json --results-folder /path/to/results_folder --project-name Project_A  --subsection-plot Testing --metrics Dice Accuracy Hausdorff Distance
        {filename} --config-files /path/to/config_file_A.json /path/to/config_file_B.json --results-folder /path/to/results_folder --project-name Project_A  --visualize-only True
        {filename} --config-files /path/to/config_file_A.json /path/to/config_file_B.json --results-folder /path/to/results_folder --project-name Project_A  --visualize-only True  --save-png True
        {filename} --config-files /path/to/config_file_A.json /path/to/config_file_B.json --results-folder /path/to/results_folder --project-name Project_A  --display-in-browser True  --save-png True
        {filename} --config-files /path/to/config_file_A.json /path/to/config_file_B.json --results-folder /path/to/results_folder --project-name Project_A  --display-in-browser True  --df-format pickle
        {filename} --config-files /path/to/config_file_A.json /path/to/config_file_B.json --results-folder /path/to/results_folder --project-name Project_A  --upload-visdom-server True  --save-png True
    """.format(  # noqa: E501 W291
        filename=Path(__file__).name
    )
)


def get_arg_parser():
    pars = ArgumentParser(description=DESC, epilog=EPILOG, formatter_class=RawTextHelpFormatter)

    pars.add_argument(
        "--config-files",
        type=str,
        required=True,
        nargs="+",
        help="Sequence of configuration files, used to identify the experiments to include in the project. ",
    )

    pars.add_argument(
        "--results-folder",
        type=str,
        required=True,
        help="Folder path where to save tables, plots and log files. ",
    )

    pars.add_argument(
        "--project-name",
        type=str,
        required=True,
        help="String used to identify the Project Name. ",
    )

    pars.add_argument(
        "--project-description",
        type=str,
        required=False,
        default="",
        help="Optional brief description of the project. ",
    )

    pars.add_argument(
        "--metrics",
        type=str,
        required=False,
        nargs="+",
        help="Sequence of metrics to be computed. If specified, the metrics listed in the configuration files are overridden",
    )

    pars.add_argument(
        "--subsection-plot",
        type=str,
        required=False,
        default=None,
        help="Option to plot only a subsection of the project results. Values can be: [ ``Validation``, ``Testing`` ]",
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
        "--upload-visdom-server",
        type=str2bool,
        required=False,
        default=False,
        help="Specify to upload the metric **Plotly** plots in the running **Visdom** server",
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

    config_files = args["config_files"]
    df_format = args["df_format"]

    experiments = []
    for config_file in config_files:
        with open(config_file) as json_file:
            config_dict = json.load(json_file)

        experiments.append(config_dict["Experiment Name"])

    metrics = []

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

    results_folder = args["results_folder"]
    Path(results_folder).joinpath(METRICS_FOLDER_NAME).mkdir(parents=True, exist_ok=True)
    summary_dict = {
        "ProjectName": args["project_name"],
        "Description": args["project_description"],
        "Experiments": experiments,
        "Metrics": metrics,
        "results_folder": results_folder,
    }
    if args["visualize_only"] is not True:
        with open(Path(results_folder).joinpath(METRICS_FOLDER_NAME, "project_summary.json"), "w") as outfile:
            json.dump(summary_dict, outfile)
        create_dataframe_for_project(results_folder, args["project_name"], config_files, metrics, df_format)

    df_paths = get_saved_dataframes(summary_dict, metrics, ["project"], df_format)
    if (
        args["save_png"] is True
        or args["save_json"] is True
        or args["save_html"] is True
        or args["show_in_browser"] is True
        or args["upload_visdom_server"] is True
    ):
        subsection = args["subsection_plot"]
        plot_dict = create_plots_for_project(
            summary_dict, df_paths, metrics, summary_dict["ProjectName"], subsection, args["plot_phase"]
        )
        if args["save_png"] is True:
            save_plots(results_folder, plot_dict, metrics, ["project"], "png")

        if args["save_json"] is True:
            save_plots(results_folder, plot_dict, metrics, ["project"], "json")

        if args["show_in_browser"] is True:
            for plot in plot_dict:
                plot_dict[plot].show()

        if args["save_html"] is True:
            save_plots(results_folder, plot_dict, metrics, ["project"], "html")

        if args["upload_visdom_server"] is True:
            for metric in metrics:
                for plot in PLOTS:
                    if plot == "bar":
                        for aggr in BAR_AGGREGATORS:
                            vis.plotlyplot(plot_dict["{}_{}_project_{}".format(aggr, metric, plot)], env=metric)
                    else:
                        vis.plotlyplot(plot_dict["{}_project_{}".format(metric, plot)], env=metric)
                create_log_at(
                    Path(results_folder).joinpath(METRICS_FOLDER_NAME, "project", metric, metric + ".log"),
                    metric,
                )


if __name__ == "__main__":
    main()
