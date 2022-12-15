#!/usr/bin/env python

import json
import os
import subprocess
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from textwrap import dedent

from Hive.utils.log_utils import get_logger, add_verbosity_options_to_argparser, log_lvl_from_verbosity_args, str2bool

DESC = dedent(
    """
    Run ``nndet_train`` command to start nnDetection training for the specified fold.
    """  # noqa: E501
)
EPILOG = dedent(
    """
    Example call:
    ::
        {filename} --config-file ../CONFIG_FILE.json --run-fold 0
        {filename} --config-file ../CONFIG_FILE.json --run-fold 0 --resume-training y
    """.format(  # noqa: E501
        filename=Path(__file__).stem
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
        type=str,
        default="0",
        required=False,
        help="Index indicating which fold to run. Default: ``0``",
    )

    pars.add_argument(
        "--resume-training",
        type=str2bool,
        default="no",
        help="Flag to indicate training resume after stopping it. Default ``no``.",
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
            "nndet_train",
            data["Task_ID"],
            "-o",
            "exp.fold={}".format(args["run_fold"]),
        ]

        if args["resume_training"]:
            arguments.append("train.mode=resume")

        os.environ["det_data"] = data["base_folder"]
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["det_num_threads"] = os.environ["N_THREADS"]
        os.environ["nnUNet_def_n_proc"] = os.environ["N_THREADS"]
        os.environ["det_models"] = data["results_folder"]

        if "receiver_email" in data:
            os.environ["receiver_email"] = data["receiver_email"]

        if int(args["run_fold"]) >= 0:
            subprocess.run(arguments)
            subprocess.run(["nndet_sweep", data["Task_ID"], "RetinaUNetV001_D3V001_3d", str(args["run_fold"])])

        # subprocess.run(
        #    ["nndet_eval", data["Task_ID"], "RetinaUNetV001_D3V001_3d", str(args['run_fold']), "--boxes", "--seg",
        #     "--analyze_boxes"])
        else:
            subprocess.run(["nndet_consolidate", data["Task_ID"], "RetinaUNetV001_D3V001_3d", "--sweep_boxes"])
            subprocess.run(
                ["nndet_seg2nii", data["Task_ID"], "RetinaUNetV001_D3V001_3d", "--fold", str(args["run_fold"])])
            subprocess.run(
                ["nndet_boxes2nii", data["Task_ID"], "RetinaUNetV001_D3V001_3d", "--fold", str(args["run_fold"])])


if __name__ == "__main__":
    main()
