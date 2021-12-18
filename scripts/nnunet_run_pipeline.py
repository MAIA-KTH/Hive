#!/usr/bin/env python

import datetime
import importlib.resources
import json
import os
import subprocess
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from textwrap import dedent

import Hive.configs
from Hive.utils.log_utils import (
    get_logger,
    add_verbosity_options_to_argparser,
    log_lvl_from_verbosity_args,
)

TIMESTAMP = "{:%Y-%m-%d_%H-%M-%S}".format(datetime.datetime.now())

DESC = dedent(
    """
    Run nnUNet pipeline:
        Data and folder preparation -> Preprocessing -> Training -> Testing set Prediction -> Testing Set Metric Evaluation.

    """  # noqa: E501
)
EPILOG = dedent(
    """
    Example call:
    ::
        {filename} -i /path/to/input_data_folder --config-file LungLobeSeg_nnUNet_3D_config.json
        {filename} -i /path/to/input_data_folder --config-file LungLobeSeg_nnUNet_3D_config.json  --task-ID 106 --test-split 25
    """.format(  # noqa: E501
        filename=Path(__file__).name
    )
)


def get_arg_parser():
    pars = ArgumentParser(description=DESC, epilog=EPILOG, formatter_class=RawTextHelpFormatter)

    pars.add_argument(
        "-i",
        "--input-data-folder",
        type=str,
        required=True,
        help="Input Dataset folder",
    )

    pars.add_argument(
        "--cv-data-folder",
        type=str,
        required=False,
        default=None,
        help="Cross-Validation Dataset folder",
    )

    pars.add_argument(
        "--task-ID",
        type=str,
        default="100",
        help="Task ID used in the folder path tree creation (Default: 100)",
    )

    pars.add_argument(
        "--test-split",
        type=int,
        choices=range(0, 101),
        metavar="[0-100]",
        default=20,
        help="Split value ( in %% ) to create Test set from Dataset (Default: 20)",
    )

    pars.add_argument(
        "--config-file",
        type=str,
        required=True,
        default="LungLobeSeg_nnUNet_3D_config.json",
        help="Configuration JSON file with experiment and dataset parameters (Default: LungLobeSeg_nnUNet_3D_config.json)",
    )

    add_verbosity_options_to_argparser(pars)

    return pars


def run_data_and_folder_preparation_step(arguments):
    try:
        with open(arguments["config_file"]) as json_file:
            config_dict = json.load(json_file)
    except FileNotFoundError:
        with importlib.resources.path(Hive.configs, arguments["config_file"]) as json_path:
            with open(json_path) as json_file:
                config_dict = json.load(json_file)

    args = [
        "nnunet_3d_prepare_data_folder.py",
        "--input-data-folder",
        arguments["input_data_folder"],
        "--task-ID",
        arguments["task_ID"],
        "--task-name",
        config_dict["DatasetName"] + "_" + config_dict["Experiment Name"],
        "--config-file",
        arguments["config_file"],
    ]

    if "cascade_step" in arguments:
        args.append("--cascade-step")
        args.append(arguments["cascade_step"])

    return args


def run_cross_validation(arguments, config_file, folds):
    arg_list = []
    for fold in folds:
        args = [
            "nnunet_run_cross_validation.py",
            "--config-file",
            config_file,
            "--run-fold",
            str(fold),
            "-i",
            arguments["cv_data_folder"],
        ]  # "--run-validation-only", "y"
        arg_list.append(args)
    return arg_list


def run_preprocessing_step(config_file):
    args = [
        "nnunet_run_preprocessing.py",
        "--config-file",
        config_file,
        "--verify_dataset_integrity",
        "-pl2d",
        "None",
        "-tl",
        os.environ["N_THREADS"],
        "-tf",
        os.environ["N_THREADS"],
    ]
    return args


def run_training_step(config_file, folds, cascade_step=None):
    arg_list = []
    for fold in folds:
        args = [
            "nnunet_run_training.py",
            "--config-file",
            config_file,
            "--run-fold",
            str(fold),
            "--npz",
        ]  # "--run-validation-only", "y"
        if cascade_step is not None:
            args.append("--cascade-step")
            args.append(cascade_step)
        arg_list.append(args)
    return arg_list


def main():
    parser = get_arg_parser()

    arguments = vars(parser.parse_args())

    logger = get_logger(  # noqa: F841
        name=Path(__file__).name,
        level=log_lvl_from_verbosity_args(arguments),
    )
    try:
        with open(arguments["config_file"]) as json_file:
            config_dict = json.load(json_file)
    except FileNotFoundError:
        with importlib.resources.path(Hive.configs, arguments["config_file"]) as json_path:
            with open(json_path) as json_file:
                config_dict = json.load(json_file)

    output_json_config_filename = (
        config_dict["DatasetName"] + "_" + config_dict["Experiment Name"] + "_" + arguments["task_ID"] + ".json"
    )
    os.environ["RESULTS_FOLDER"] = str(
        Path(os.environ["root_experiment_folder"]).joinpath(
            config_dict["Experiment Name"], config_dict["Experiment Name"] + "_results"
        )
    )
    output_json_config_file = str(Path(os.environ["RESULTS_FOLDER"]).joinpath(output_json_config_filename))

    cascade_steps = 1
    if "Cascade_steps" in config_dict:
        cascade_steps = config_dict["Cascade_steps"]

    pipeline_steps = []
    for cascade_step in range(cascade_steps):
        if cascade_steps > 1:
            arguments["cascade_step"] = str(cascade_step)

        pipeline_steps.append(run_data_and_folder_preparation_step(arguments))
        pipeline_steps.append(run_preprocessing_step(output_json_config_file))
        if cascade_steps > 1:
            [
                pipeline_steps.append(step)
                for step in run_training_step(output_json_config_file, range(config_dict["n_folds"]), str(cascade_step))
            ]
        else:
            [pipeline_steps.append(step) for step in run_training_step(output_json_config_file, range(config_dict["n_folds"]))]
        if arguments["cv_data_folder"] is not None:
            [
                pipeline_steps.append(step)
                for step in run_cross_validation(arguments, output_json_config_file, range(config_dict["n_folds"]))
            ]
        # "Hive_evaluate_predictions.py"
        # "Hive_generate_experiment_results.py"
    # "nnunet_run_prediction.py"
    # "Hive_evaluate_predictions.py"
    # "Hive_generate_experiment_results.py"
    Path(os.environ["root_experiment_folder"]).joinpath(config_dict["Experiment Name"]).mkdir(exist_ok=True, parents=True)
    pipeline_steps_summary = open(
        Path(os.environ["root_experiment_folder"]).joinpath(
            config_dict["Experiment Name"], "Task_" + arguments["task_ID"] + "_" + TIMESTAMP + ".txt"
        ),
        "w",
    )
    for step in pipeline_steps:
        pipeline_steps_summary.write(" ".join(step) + "\n")

    pipeline_steps_summary.close()

    for step in pipeline_steps:
        subprocess.run(step)


if __name__ == "__main__":
    main()
