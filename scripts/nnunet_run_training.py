#!/usr/bin/env python

import json
import os
import subprocess
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from textwrap import dedent

from Hive.utils.data_folder_utils import order_data_folder_by_patient, order_data_in_single_folder
from Hive.utils.log_utils import get_logger, add_verbosity_options_to_argparser, log_lvl_from_verbosity_args, str2bool

DESC = dedent(
    """
    Run nnUNet command to start training for the specified fold.
    """  # noqa: E501
)
EPILOG = dedent(
    """
    Example call:
    ::
        {filename} --config-file ../LungLobeSeg_3d_fullres_100_LungLobeSeg_3D_Single_Modality.json --run-fold 0
        {filename} --config-file ../LungLobeSeg_3d_fullres_100_LungLobeSeg_3D_Single_Modality.json --run-fold 0 --npz
    """.format(  # noqa: E501
        filename=Path(__file__).name
    )
)


def get_arg_parser():
    pars = ArgumentParser(description=DESC, epilog=EPILOG, formatter_class=RawTextHelpFormatter)

    pars.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="File path for the configuration dictionary, used to retrieve experiments variables (Task_ID)",
    )

    pars.add_argument(
        "--run-fold",
        type=int,
        choices=range(0, 5),
        metavar="[0-4]",
        default=0,
        help="int value indicating which fold (in the range 0-4) to run",
    )

    pars.add_argument(
        "--run-validation-only",
        type=str2bool,
        default="n",
        help="Flag to run only validation part.",
    )

    pars.add_argument(
        "--run-task",
        type=str,
        default=None,
        help="Optional parameter to select the Primary Task when running Multi-Tasking training",
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
        data = json.load(json_file)

        arguments = [
            "nnUNet_train",
            data["TRAINING_CONFIGURATION"],
            data["TRAINER_CLASS_NAME"],
            "Task" + data["Task_ID"] + "_" + data["Task_Name"],
            str(args["run_fold"]),
        ]
        if args["run_validation_only"]:
            arguments.append("-val")
        if args["run_task"] is not None:
            arguments.append("--val_folder")
            arguments.append("validation_raw_{}".format(args["run_task"]))
            arguments.append("--run-task")
            arguments.append(args["run_task"])
        arguments.extend(unknown_arguments)

        os.environ["nnUNet_raw_data_base"] = data["base_folder"]
        os.environ["nnUNet_preprocessed"] = data["preprocessing_folder"]
        os.environ["RESULTS_FOLDER"] = data["results_folder"]
        os.environ["nnUNet_def_n_proc"] = os.environ["N_THREADS"]
        os.environ["MKL_THREADING_LAYER"] = "GNU"
        os.environ["nnunet_use_progress_bar"] = "1"
        if "receiver_email" in data:
            os.environ["receiver_email"] = data["receiver_email"]

        subprocess.run(arguments)
        if "--sub-step" in arguments:
            sub_step = arguments[arguments.index("--sub-step") + 1]
            fold_path = str(
                Path(data["predictions_{}_path".format(sub_step)]).joinpath(
                    "fold_{}".format(args["run_fold"]), data["predictions_folder_name"]
                )
            )
        elif args["run_task"] is not None:
            fold_path = str(
                Path(data["predictions_path"]).joinpath(
                    "fold_{}".format(args["run_fold"]), "validation_raw_{}_postprocessed".format(args["run_task"])
                )
            )
        else:
            fold_path = str(
                Path(data["predictions_path"]).joinpath("fold_{}".format(args["run_fold"]), data["predictions_folder_name"])
            )

        order_data_in_single_folder(fold_path, fold_path)
        order_data_folder_by_patient(fold_path, data["FileExtension"])
        if "--sub-step" in arguments:
            sub_step = arguments[arguments.index("--sub-step") + 1]
            if sub_step == "step_1":
                cascade_folder_path = str(
                    Path(data["predictions_path"]).joinpath("fold_{}".format(args["run_fold"]), data["predictions_folder_name"])
                )
                order_data_in_single_folder(cascade_folder_path, cascade_folder_path)
                order_data_folder_by_patient(cascade_folder_path, data["FileExtension"])


if __name__ == "__main__":
    main()
