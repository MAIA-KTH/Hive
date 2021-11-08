#!/usr/bin/env python

import json
import os
import subprocess
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from textwrap import dedent

from Hive.utils.file_utils import move_file_in_subfolders
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
        arguments.extend(unknown_arguments)

        os.environ["nnUNet_raw_data_base"] = data["base_folder"]
        os.environ["nnUNet_preprocessed"] = data["preprocessing_folder"]
        os.environ["RESULTS_FOLDER"] = data["results_folder"]
        os.environ["nnUNet_def_n_proc"] = os.environ["N_THREADS"]
        os.environ["MKL_THREADING_LAYER"] = "GNU"
        os.environ["nnunet_use_progress_bar"] = "1"
        os.environ["email_account"] = "simonebendazzoli93@gmail.com"
        os.environ["email_password"] = "wtdlstmbiylihyst"
        os.environ["receiver_email"] = "simonebendazzoli93@gmail.com"

        subprocess.run(arguments)
        if "--cascade-step" in arguments:
            cascade_step = arguments[arguments.index("--cascade-step") + 1]
            fold_path = str(
                Path(data["predictions_step_{}_path".format(cascade_step)]).joinpath(
                    "fold_{}".format(args["run_fold"]), data["predictions_folder_name"]
                )
            )
        else:
            fold_path = str(
                Path(data["predictions_path"]).joinpath("fold_{}".format(args["run_fold"]), data["predictions_folder_name"])
            )
        move_file_in_subfolders(fold_path, data["FileExtension"], data["FileExtension"])


if __name__ == "__main__":
    main()
