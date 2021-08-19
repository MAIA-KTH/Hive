#!/usr/bin/env python

import json
import os
from pathlib import Path
from argparse import ArgumentParser, RawTextHelpFormatter
from textwrap import dedent

from k8s_DP.utils.log_utils import (
    get_logger,
    add_verbosity_options_to_argparser,
    log_lvl_from_verbosity_args,
)

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
            data["TRAINING_CONFIGURATION"],
            data["TRAINER_CLASS_NAME"],
            "Task" + data["Task_ID"] + "_" + data["Task_Name"],
            str(args["run_fold"]),
        ]
        arguments.extend(unknown_arguments)
        os.system("nnUNet_train " + " ".join(arguments))


if __name__ == "__main__":
    main()
