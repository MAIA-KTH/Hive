#!/usr/bin/env python

import json
import os
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from textwrap import dedent

import numpy as np
from sklearn.model_selection import KFold

from Hive.utils.file_utils import split_dataset, copy_data_to_dataset_folder
from Hive.utils.log_utils import (
    get_logger,
    add_verbosity_options_to_argparser,
    log_lvl_from_verbosity_args,
)

DESC = dedent(
    """
    Script to run Cross-Validation for a given trained model on a different (from the training) dataset.
    """  # noqa: E501
)
EPILOG = dedent(
    """
      Example call:
    ::
        {filename} -i /path/to/cv_data_folder --config-file /path/to/config_file.json
        {filename} -i /path/to/cv_data_folder --config-file /path/to/config_file.json --run-fold 1
    """.format(  # noqa: E501
        filename=Path(__file__).name
    )
)


def get_arg_parser():
    pars = ArgumentParser(description=DESC, epilog=EPILOG, formatter_class=RawTextHelpFormatter)

    pars.add_argument(
        "-i",
        "--input-folder",
        type=str,
        required=True,
        help="Folder path containing the volumes to be cross-predicted",
    )

    pars.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="File path for the configuration dictionary, used to retrieve experiments variables.",
    )

    pars.add_argument(
        "--run-fold",
        type=str,
        default="All",
        help="Fold/subset to run for cross-validation. Can be a numeric string (to run a specific fold validation),"
        '``"test_set"``, to run the test set cross-validation, or ``"All"``, to run the cross-validation over '
        "the entire dataset.",
    )

    pars.add_argument(
        "--cv-name",
        type=str,
        required=True,
        help="String value representing the cross-validation, to be appended to the Experiment name.",
    )

    pars.add_argument(
        "--sub-steps",
        type=str,
        required=False,
        nargs="+",
        default=None,
        help="",
    )

    add_verbosity_options_to_argparser(pars)

    return pars


def main():
    parser = get_arg_parser()
    arguments, unknown_arguments = parser.parse_known_args()
    args = vars(arguments)

    logger = get_logger(  # NOQA: F841
        name=Path(__file__).name,
        level=log_lvl_from_verbosity_args(args),
    )

    config_file = args["config_file"]

    with open(config_file) as json_file:
        config_dict = json.load(json_file)

    os.environ["nnUNet_raw_data_base"] = config_dict["base_folder"]
    os.environ["nnUNet_preprocessed"] = config_dict["preprocessing_folder"]
    os.environ["RESULTS_FOLDER"] = config_dict["results_folder"]
    os.environ["nnUNet_def_n_proc"] = os.environ["N_THREADS"]
    os.environ["MKL_THREADING_LAYER"] = "GNU"
    os.environ["nnunet_use_progress_bar"] = "1"

    train_dataset, test_dataset = split_dataset(args["input_folder"], config_dict["train_test_split"], config_dict["Seed"])

    fold = args["run_fold"]
    cv_name = args["cv_name"]
    if fold == "test_set":
        selected_test_data = test_dataset
    elif fold == "All":
        selected_test_data = train_dataset
        _ = [selected_test_data.append(test_case) for test_case in test_dataset]
    else:
        train_dataset_sorted = np.sort(train_dataset)
        selected_test_data = []
        kfold = KFold(n_splits=config_dict["n_folds"], shuffle=True, random_state=12345)  # self.config_dict["Seed"])
        for i, (train_idx, test_idx) in enumerate(kfold.split(train_dataset_sorted)):
            if i == fold:
                for test in test_idx:
                    selected_test_data.append(train_dataset_sorted[test])

    output_base_folder_list = []
    if args["sub_steps"] is None:
        output_base_folder = Path(os.environ["root_experiment_folder"]).joinpath(
            config_dict["Experiment Name"],
            config_dict["Experiment Name"] + "_cross_validation",
            config_dict["Experiment Name"] + "_to_{}".format(cv_name),
            config_dict["Experiment Name"] + "_to_{}_base".format(cv_name),
            config_dict["Experiment Name"] + "_to_{}_raw_data".format(cv_name),
            "Task"
            + config_dict["Task_ID"]
            + "_"
            + config_dict["DatasetName"]
            + "_"
            + config_dict["Experiment Name"]
            + "_to_{}".format(cv_name),
        )
        output_base_folder.mkdir(parents=True, exist_ok=True)

        output_base_folder.joinpath("imagesTs").mkdir(parents=True, exist_ok=True)
        output_base_folder.joinpath("labelsTs").mkdir(parents=True, exist_ok=True)

        copy_data_to_dataset_folder(
            args["input_folder"],
            selected_test_data,
            str(output_base_folder),
            "imagesTs",
            config_dict,
            "labelsTs",
        )

        output_results_folder = Path(os.environ["root_experiment_folder"]).joinpath(
            config_dict["Experiment Name"],
            config_dict["Experiment Name"] + "_cross_validation",
            config_dict["Experiment Name"] + "_to_{}".format(cv_name),
            config_dict["Experiment Name"] + "_to_{}_results".format(cv_name),
            "nnUNet",
            config_dict["TRAINING_CONFIGURATION"],
            "Task"
            + config_dict["Task_ID"]
            + "_"
            + config_dict["DatasetName"]
            + "_"
            + config_dict["Experiment Name"]
            + "_to_{}".format(cv_name),
            config_dict["TRAINER_CLASS_NAME"] + "__" + config_dict["TRAINER_PLAN"],
        )

        output_results_folder.mkdir(parents=True, exist_ok=True)

        if fold == "test_set" or fold == "All":
            output_fold_results_folder = output_results_folder.joinpath(config_dict["predictions_folder_name"])

        else:
            output_fold_results_folder = output_results_folder.joinpath(
                "fold_{}".format(fold), config_dict["predictions_folder_name"]
            )

        output_fold_results_folder.mkdir(parents=True, exist_ok=True)
    else:
        for sub_step in args["sub_steps"]:
            output_base_folder = Path(os.environ["root_experiment_folder"]).joinpath(
                config_dict["Experiment Name"],
                config_dict["Experiment Name"] + "_cross_validation",
                config_dict["Experiment Name"] + "_to_{}".format(cv_name),
                config_dict["Experiment Name"] + "_to_{}_base".format(cv_name),
                config_dict["Experiment Name"] + "_to_{}_raw_data".format(cv_name),
                "Task"
                + config_dict["Task_ID"]
                + "_"
                + config_dict["DatasetName"]
                + "_"
                + config_dict["Experiment Name"]
                + "_to_{}".format(cv_name),
                sub_step,
            )
            output_base_folder.mkdir(parents=True, exist_ok=True)
            output_base_folder_list.append(output_base_folder)
            output_base_folder.joinpath("imagesTs").mkdir(parents=True, exist_ok=True)
            output_base_folder.joinpath("labelsTs").mkdir(parents=True, exist_ok=True)

            config_dict["label_suffix"] = config_dict[sub_step]["label_suffix"]
            config_dict["label_dict"] = config_dict[sub_step]["label_dict"]
            config_dict["Modalities"] = config_dict[sub_step]["Modalities"]

            copy_data_to_dataset_folder(
                args["input_folder"],
                selected_test_data,
                str(output_base_folder),
                "imagesTs",
                config_dict,
                "labelsTs",
            )

        output_results_folder = Path(os.environ["root_experiment_folder"]).joinpath(
            config_dict["Experiment Name"],
            config_dict["Experiment Name"] + "_cross_validation",
            config_dict["Experiment Name"] + "_to_{}".format(cv_name),
            config_dict["Experiment Name"] + "_to_{}_results".format(cv_name),
            "nnUnet",
            "cascade",
            "Task"
            + config_dict["Task_ID"]
            + "_"
            + config_dict["DatasetName"]
            + "_"
            + config_dict["Experiment Name"]
            + "_to_{}".format(cv_name),
            config_dict["TRAINER_CLASS_NAME"] + "__" + config_dict["TRAINER_PLAN"],
        )

        output_results_folder.mkdir(parents=True, exist_ok=True)

        if fold == "test_set" or fold == "All":
            output_fold_results_folder = output_results_folder.joinpath(config_dict["predictions_folder_name"])

        else:
            output_fold_results_folder = output_results_folder.joinpath(
                "fold_{}".format(fold), config_dict["predictions_folder_name"]
            )

        output_fold_results_folder.mkdir(parents=True, exist_ok=True)

    original_experiment_name = config_dict["Experiment Name"]
    config_dict["dataset_folder"] = Path(args["input_folder"]).stem.replace("_", " ")
    config_dict["Experiment Name"] = config_dict["Experiment Name"] + "_to_{}".format(cv_name)
    original_task_name = config_dict["Task_Name"]
    config_dict["Task_Name"] = config_dict["Task_Name"] + "_to_{}".format(cv_name)
    config_dict["results_folder"] = str(
        Path(os.environ["root_experiment_folder"]).joinpath(
            original_experiment_name,
            original_experiment_name + "_cross_validation",
            config_dict["Experiment Name"],
            config_dict["Experiment Name"] + "_results",
        )
    )
    config_dict["predictions_path"] = str(output_results_folder)

    output_json_filepath = Path(config_dict["results_folder"]).joinpath(
        config_dict["DatasetName"] + "_" + config_dict["Experiment Name"] + "_{}.json".format(config_dict["Task_ID"])
    )
    with open(output_json_filepath, "w") as output_json:
        json.dump(config_dict, output_json)

    if args["sub_steps"] is None:
        arguments_list = [
            "-i",
            str(output_base_folder.joinpath("imagesTs")),
            "-o",
            str(output_fold_results_folder),
            "-m",
            config_dict["TRAINING_CONFIGURATION"],
            "-t",
            "Task" + config_dict["Task_ID"] + "_" + original_task_name,
            "-tr",
            config_dict["TRAINER_CLASS_NAME"],
        ]
        if fold != "test_set" and fold != "All":
            arguments_list.append("-f")
            arguments_list.append(fold)
        arguments_list.extend(unknown_arguments)

        os.system("nnUNet_predict " + " ".join(arguments_list))
    else:
        for idx, sub_step in enumerate(args["sub_steps"]):
            output_folder = str(output_fold_results_folder)
            if idx != len(args["sub_steps"]) - 1:
                output_folder = output_folder + "_{}".format(sub_step)
            arguments_list = [
                "-i",
                str(output_base_folder_list[idx].joinpath("imagesTs")),
                "-o",
                output_folder,
                "-m",
                sub_step,
                "-t",
                "Task" + config_dict["Task_ID"] + "_" + original_task_name,
                "-tr",
                config_dict["TRAINER_CLASS_NAME"],
                "--sub-step",
                sub_step,
            ]
            if fold != "test_set" and fold != "All":
                arguments_list.append("-f")
                arguments_list.append(fold)
            arguments_list.extend(unknown_arguments)

            os.system("nnUNet_predict " + " ".join(arguments_list))


if __name__ == "__main__":
    main()
