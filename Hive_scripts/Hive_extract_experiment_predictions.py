#!/usr/bin/env python

import json
from argparse import ArgumentParser, RawTextHelpFormatter
from distutils.dir_util import copy_tree
from pathlib import Path
from textwrap import dedent

# from old_src.evaluation import get_results_summary_filepath
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

    add_verbosity_options_to_argparser(pars)
    return pars


def main():
    parser = get_arg_parser()

    args = vars(parser.parse_args())

    logger = get_logger(  # NOQA: F841
        name=Path(__file__).name,
        level=log_lvl_from_verbosity_args(args),
    )

    with open(args["config_file"]) as json_file:
        config_dict = json.load(json_file)

    output_path = args["output_experiment_folder"]

    sections = ["validation"]

    for section in sections:
        prediction_directory_out = Path(output_path).joinpath(
            "Task" + config_dict["Task_ID"] + "_{}".format(config_dict["Task_Name"]), section
        )
        prediction_directory_out.mkdir(parents=True, exist_ok=True)
        prediction_directory = Path(config_dict["results_folder"]).joinpath(
            "Task" + config_dict["Task_ID"] + "_{}".format(config_dict["Task_Name"]),
            "RetinaUNetV001_D3V001_3d",
            "consolidated",
            "val_predictions_nii",
        )
        if prediction_directory.is_dir():
            subject_predictions = subfolders(prediction_directory, join=False)
            for subject_prediction in subject_predictions:
                copy_tree(
                    str(Path(prediction_directory).joinpath(subject_prediction)),
                    str(Path(prediction_directory_out).joinpath(subject_prediction)),
                )


if __name__ == "__main__":
    main()
