#!/usr/bin/env python

import json
import os
from argparse import ArgumentParser, RawTextHelpFormatter
from distutils.dir_util import copy_tree
from pathlib import Path
from textwrap import dedent

from Hive.evaluation.io_metric_results import get_results_summary_filepath
from Hive.utils.file_utils import subfolders
from Hive.utils.log_utils import get_logger, add_verbosity_options_to_argparser, log_lvl_from_verbosity_args

DESC = dedent(
    """
    Script used to copy and save Experiment predictions, from the original experiment folder to the specified output folder.
    """  # noqa: E501 W291 W605
)
EPILOG = dedent(
    """
    Example call:
    ::
        {filename} --config-file /path/to/config_file.json --output-experiment-folder /home/Experiment_Predictions
        {filename} --config-file /path/to/config_file.json --output-experiment-folder /home/Experiment_Predictions --sections validation
        {filename} --config-file /path/to/config_file.json --output-experiment-folder /home/Experiment_Predictions --sections validation --prediction-suffix post
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
        "--output-experiment-folder",
        type=str,
        required=True,
        help="Folder path to set the output experiment folder.",
    )

    pars.add_argument(
        "--root-experiment-folder",
        type=str,
        required=False,
        help="Optional value to set the root experiment folder (if not existing or different from the env variable) ",
    )

    pars.add_argument(
        "--sections",
        type=str,
        required=False,
        nargs="+",
        help="Sequence of sections to extract the prediction files. Values can be: [ ``testing``, ``validation`` ].",
    )

    pars.add_argument(
        "--prediction-suffix",
        type=str,
        required=False,
        default="",
        help="Prediction name suffix to find the corresponding prediction folder. Defaults to ``" "`` ",
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

    output_path = args["output_experiment_folder"]

    if args["root_experiment_folder"] is not None:
        config_dict["root_results_folder"] = args["root_experiment_folder"]
    else:
        config_dict["root_results_folder"] = os.environ["root_experiment_folder"]

    if "root_results_folder" in config_dict:
        config_dict["results_folder"] = config_dict["results_folder"].replace(
            os.environ["root_experiment_folder"], config_dict["root_results_folder"]
        )
        config_dict["predictions_path"] = config_dict["predictions_path"].replace(
            os.environ["root_experiment_folder"], config_dict["root_results_folder"]
        )

    sections = ["testing", "validation"]
    n_folds = config_dict["n_folds"]
    if args["sections"]:
        sections = args["sections"]

    for fold in range(n_folds):
        for section in sections:
            json_summary = get_results_summary_filepath(config_dict, section, prediction_suffix, fold)
            prediction_parent_directory = Path(json_summary).parent
            prediction_parent_directory_out = Path(output_path).joinpath(config_dict["Experiment Name"], section)
            Path(prediction_parent_directory_out).parent.mkdir(parents=True, exist_ok=True)
            if prediction_parent_directory.is_dir():
                predictions = subfolders(prediction_parent_directory, join=False)
                for prediction_folder in predictions:
                    copy_tree(
                        str(Path(prediction_parent_directory).joinpath(prediction_folder)),
                        str(Path(prediction_parent_directory_out).joinpath(prediction_folder)),
                    )


if __name__ == "__main__":
    main()
