#!/usr/bin/env python

import csv
import datetime
import importlib.resources
import json
import os
import re
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from textwrap import dedent

import Hive.configs
import numpy as np
from Hive.utils.file_utils import (
    create_nnunet_data_folder_tree,
    split_dataset,
    copy_data_to_dataset_folder,
    save_config_json,
    generate_dataset_json,
)
from Hive.utils.log_utils import get_logger, add_verbosity_options_to_argparser, log_lvl_from_verbosity_args, INFO
from sklearn.model_selection import KFold

TIMESTAMP = "{:%Y-%m-%d_%H-%M-%S}".format(datetime.datetime.now())

DESC = dedent(
    """
    Standardize dataset for 3D LungLobeSeg experiment, to be compatible with nnUNet framework.
    The dataset is assumed to be in NIFTI format (*.nii.gz) and containing only a single
    modality ( CT ) for the input 3D volumes.

    """  # noqa: E501
)
EPILOG = dedent(
    """
    Example call:
    ::
        {filename} -i /path/to/input_data_folder --task-ID 106 --task-name LungLobeSeg3D  --config-file LungLobeSeg_nnUNet_3D_config.json
        {filename} --input /path/to/input_data_folder --task-ID 101 --task-name 3D_LungLobeSeg --test-split 30 --config-file LungLobeSeg_nnUNet_3D_config.json
    """.format(  # noqa: E501
        filename=Path(__file__).name
    )
)


def main():
    parser = get_arg_parser()

    arguments = vars(parser.parse_args())

    logger = get_logger(
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

    if "receiver_email" not in os.environ:
        try:
            user_email = input("Please Enter a valid e-mail to receive experiments updates:\n")
            regex = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
            if re.fullmatch(regex, user_email):
                logger.log(INFO, "Using {} for e-mail updates.".format(user_email))
                config_dict["receiver_email"] = user_email
            else:
                logger.log(INFO, "{} not valid, disabling e-mail updates.".format(user_email))
        except KeyboardInterrupt:
            logger.log(INFO, "Disabling e-mail updates.")

    os.environ["raw_data_base"] = str(
        Path(os.environ["root_experiment_folder"]).joinpath(
            config_dict["Experiment Name"], config_dict["Experiment Name"] + "_base"
        )
    )
    os.environ["preprocessed_folder"] = str(
        Path(os.environ["root_experiment_folder"]).joinpath(
            config_dict["Experiment Name"], config_dict["Experiment Name"] + "_preprocess"
        )
    )
    os.environ["RESULTS_FOLDER"] = str(
        Path(os.environ["root_experiment_folder"]).joinpath(
            config_dict["Experiment Name"], config_dict["Experiment Name"] + "_results"
        )
    )
    try:
        dataset_path = str(
            Path(os.environ["raw_data_base"]).joinpath(
                "nnUNet_raw_data",
                "Task" + arguments["task_ID"] + "_" + arguments["task_name"],
            )
        )

    except KeyError:
        logger.error("raw_data_base is not set as environment variable")
        return 1

    create_nnunet_data_folder_tree(
        os.environ["raw_data_base"],
        arguments["task_name"],
        arguments["task_ID"],
    )
    train_dataset, test_dataset = split_dataset(arguments["input_data_folder"], arguments["test_split"], config_dict["Seed"])

    dataset_split = []
    for test_subject in test_dataset:
        dataset_split_dict = {"Subject": test_subject, "Split": "Testing"}
        dataset_split.append(dataset_split_dict)

    train_dataset_sorted = np.sort(train_dataset)
    kfold = KFold(n_splits=config_dict["n_folds"], shuffle=True, random_state=12345)  # self.config_dict["Seed"])
    for i, (train_idx, test_idx) in enumerate(kfold.split(train_dataset_sorted)):
        for test in test_idx:
            dataset_split_dict = {"Subject": train_dataset_sorted[test], "Split": "Validation_fold_{}".format(i)}
            dataset_split.append(dataset_split_dict)

    dataset_split_summary = Path(os.environ["root_experiment_folder"]).joinpath(
        config_dict["Experiment Name"], "dataset_split.csv"
    )

    with open(dataset_split_summary, "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["Subject", "Split"])
        writer.writeheader()
        for data in dataset_split:
            writer.writerow(data)

    config_dict["dataset_folder"] = Path(arguments["input_data_folder"]).stem.replace("_", " ")

    if "Cascade_steps" in config_dict:
        cascade_step = arguments["sub_step"]
        dataset_path = str(Path(dataset_path).joinpath(cascade_step))
        create_nnunet_data_folder_tree(os.environ["raw_data_base"], arguments["task_name"], arguments["task_ID"], cascade_step)
        config_dict["label_suffix"] = config_dict[cascade_step]["label_suffix"]
        config_dict["label_dict"] = config_dict[cascade_step]["label_dict"]
        config_dict["Modalities"] = config_dict[cascade_step]["Modalities"]

    copy_data_to_dataset_folder(
        arguments["input_data_folder"],
        train_dataset,
        dataset_path,
        "imagesTr",
        config_dict,
        "labelsTr",
    )
    copy_data_to_dataset_folder(
        arguments["input_data_folder"],
        test_dataset,
        dataset_path,
        "imagesTs",
        config_dict,
        "labelsTs",
    )
    n_tasks = 1
    if type(config_dict["label_suffix"]) == list:
        n_tasks = len(config_dict["label_suffix"])

    generate_dataset_json(
        str(Path(dataset_path).joinpath("dataset.json")),
        str(Path(dataset_path).joinpath("imagesTr")),
        str(Path(dataset_path).joinpath("imagesTs")),
        list(config_dict["Modalities"].values()),
        config_dict["label_dict"],
        config_dict["DatasetName"],
        n_tasks=n_tasks,
    )

    config_dict["Task_ID"] = arguments["task_ID"]
    config_dict["Task_Name"] = arguments["task_name"]
    config_dict["train_test_split"] = arguments["test_split"]
    config_dict["base_folder"] = os.environ["raw_data_base"]

    output_json_basename = config_dict["Task_Name"] + "_" + config_dict["Task_ID"] + ".json"

    try:
        full_task_name = "Task" + config_dict["Task_ID"] + "_" + config_dict["Task_Name"]
        config_dict["results_folder"] = os.environ["RESULTS_FOLDER"]
        config_dict["predictions_path"] = str(
            Path(os.environ["RESULTS_FOLDER"]).joinpath(
                "nnUNet",
                config_dict["TRAINING_CONFIGURATION"],
                full_task_name,
                config_dict["TRAINER_CLASS_NAME"] + "__" + config_dict["TRAINER_PLAN"],
            )
        )
        if "Cascade_steps" in config_dict:
            config_dict["predictions_path"] = str(
                Path(os.environ["RESULTS_FOLDER"]).joinpath(
                    "nnUNet",
                    "cascade",
                    full_task_name,
                    config_dict["TRAINER_CLASS_NAME"] + "__" + config_dict["TRAINER_PLAN"],
                )
            )
            for cascade_step in config_dict["Cascade_steps"]:
                config_dict["predictions_{}_path".format(cascade_step)] = str(
                    Path(os.environ["RESULTS_FOLDER"]).joinpath(
                        "nnUNet",
                        cascade_step,
                        full_task_name,
                        config_dict["TRAINER_CLASS_NAME"] + "__" + config_dict["TRAINER_PLAN"],
                    )
                )
        Path(config_dict["results_folder"]).mkdir(parents=True, exist_ok=True)
    except KeyError:
        logger.warning("RESULTS_FOLDER is not set as environment variable, {} is not saved".format(output_json_basename))
        return 1
    try:
        config_dict["preprocessing_folder"] = os.environ["preprocessed_folder"]
        Path(config_dict["preprocessing_folder"]).mkdir(parents=True, exist_ok=True)

    except KeyError:
        logger.warning(
            "preprocessed_folder is not set as environment variable, not saved in {}".format(output_json_basename)
            # noqa E501
        )
    save_config_json(config_dict, str(Path(config_dict["results_folder"]).joinpath(output_json_basename)))


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
        "--task-ID",
        type=str,
        default="100",
        help="Task ID used in the folder path tree creation (Default: 100)",
    )

    pars.add_argument(
        "--task-name",
        type=str,
        required=True,
        help="Task Name used in the folder path tree creation.",  # noqa E501
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
        help="Configuration JSON file with experiment and dataset parameters.",
    )

    pars.add_argument(
        "--sub-step",
        type=str,
        required=False,
        default=None,
        help="Optional selected sub-step, to run the script in multi-step contexts(Cascade/2.5D).",
    )

    add_verbosity_options_to_argparser(pars)

    return pars


if __name__ == "__main__":
    main()