#!/usr/bin/env python

import json
import os
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from textwrap import dedent

from Hive.utils.log_utils import (
    get_logger,
    add_verbosity_options_to_argparser,
    log_lvl_from_verbosity_args,
)

DESC = dedent(
    """
    Run nnUNet command to create experiment plan, preprocessing and ( optionally ) verify the dataset integrity.

    """  # noqa: E501
)
EPILOG = dedent(
    """
    Example call:
    ::
        {filename} --config-file ../LungLobeSeg_3d_fullres_100_LungLobeSeg_3D_Single_Modality.json
        {filename} --config-file ../LungLobeSeg_3d_fullres_100_LungLobeSeg_3D_Single_Modality.json --verify_dataset_integrity
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
        help="File path for the configuration dictionary, used to retrieve experiments variables (Task_ID) ",
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
            "-t",
            data["Task_ID"],
        ]

        os.environ["nnUNet_raw_data_base"] = data["base_folder"]
        os.environ["nnUNet_preprocessed"] = data["preprocessing_folder"]
        os.environ["nnUNet_def_n_proc"] = os.environ["N_THREADS"]
        os.environ["RESULTS_FOLDER"] = data["results_folder"]
        arguments.extend(unknown_arguments)
        os.system("nnUNet_plan_and_preprocess " + " ".join(arguments))


if __name__ == "__main__":
    main()
