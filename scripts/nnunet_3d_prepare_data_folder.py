#!/usr/bin/env python

import datetime
import importlib.resources
import json
import os
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from textwrap import dedent

import Hive.configs
from Hive.utils.file_utils import (
    create_nnunet_data_folder_tree,
    split_dataset,
    copy_data_to_dataset_folder,
    save_config_json,
    generate_dataset_json,
)
from Hive.utils.log_utils import (
    get_logger,
    add_verbosity_options_to_argparser,
    log_lvl_from_verbosity_args,
)

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
        dataset_path = str(
            Path(os.environ["raw_data_base"]).joinpath(
                "nnUNet_raw_data",
                "Task" + arguments["task_ID"] + "_" + arguments["task_name"],
            )
        )

    except KeyError:
        logger.error("raw_data_base is not set as environment variable")
        return 1

    try:
        with open(arguments["config_file"]) as json_file:
            config_dict = json.load(json_file)
    except FileNotFoundError:
        with importlib.resources.path(Hive.configs, arguments["config_file"]) as json_path:
            with open(json_path) as json_file:
                config_dict = json.load(json_file)

    create_nnunet_data_folder_tree(
        os.environ["raw_data_base"],
        arguments["task_name"],
        arguments["task_ID"],
    )
    train_dataset, test_dataset = split_dataset(arguments["input_data_folder"], arguments["test_split"], config_dict["Seed"])
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
    generate_dataset_json(
        str(Path(dataset_path).joinpath("dataset.json")),
        str(Path(dataset_path).joinpath("imagesTr")),
        str(Path(dataset_path).joinpath("imagesTs")),
        list(config_dict["Modalities"].values()),
        config_dict["label_dict"],
        config_dict["DatasetName"],
    )

    config_dict["Task_ID"] = arguments["task_ID"]
    config_dict["Task_Name"] = arguments["task_name"]
    config_dict["base_folder"] = os.environ["raw_data_base"]

    output_json_basename = (
        config_dict["DatasetName"]
        + "_"
        + config_dict["TRAINING_CONFIGURATION"]
        + "_"
        + config_dict["Task_ID"]
        + "_"
        + config_dict["Task_Name"]
        + ".json"
    )

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

    add_verbosity_options_to_argparser(pars)

    return pars


if __name__ == "__main__":
    main()
