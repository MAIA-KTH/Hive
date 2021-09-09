#!/usr/bin/env python

import argparse
import json
import logging
import sys
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from textwrap import dedent

import monai
from Hive.evaluation.evaluator import compute_metrics_and_save_json
from Hive.monai.engines.evaluator import HiveEnsembled2Dto3DEvaluator
from Hive.monai.transforms.model_transforms import post_processing_25D_transform, pre_processing_25D_transform
from Hive.utils.file_utils import subfiles
from Hive.utils.log_utils import get_logger, add_verbosity_options_to_argparser, log_lvl_from_verbosity_args
from monai.data import list_data_collate

DESC = dedent(
    """
    Run 2.5D Ensembled Inference on all the files saved in **input-folder**, saving the results in **output-folder**.
    If corresponding ground truth files are available, ``gt-folder`` can be set to specify the folder path and run the
    score metrics evaluation.
    """  # noqa: W291
)

EPILOG = dedent(
    """
    Example call:
    ::
        {filename} --input-folder /path/to/input_files --output-folder /path/to/output --config-file /path/to/config_file.json 
        {filename} --input-folder /path/to/input_files --output-folder /path/to/output --config-file /path/to/config_file.json  --gt-folder /path/to/gt_files
    """.format(  # noqa: W291 E501
        filename=Path(__file__).name
    )
)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_arg_parser():
    pars = ArgumentParser(description=DESC, epilog=EPILOG, formatter_class=RawTextHelpFormatter)

    pars.add_argument(
        "--input-folder",
        type=str,
        required=True,
        help="Folder path containing the files to run the ensembled inference on. ",
    )

    pars.add_argument(
        "--output-folder",
        type=str,
        required=True,
        help="Folder path where to save the predictions as NIFTI files. ",
    )

    pars.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="File path for the configuration dictionary, used to retrieve experiment settings ",
    )

    pars.add_argument(
        "--gt-folder",
        type=str,
        required=False,
        default=None,
        help="Optional Ground Truth folder where to find the gt files and run the metric scores evaluation, saving the"
        "results in a JSON file. ",
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

    input_folder = args["input_folder"]
    output_folder = args["output_folder"]

    with open(args["config_file"]) as json_file:
        config_dict = json.load(json_file)

    results_folder = config_dict["results_folder"]
    prediction_suffix = config_dict["post_processing_suffix"]

    gt_folder = args["gt_folder"]

    run_25d_ensembled_evaluation_for_folder(input_folder, config_dict, output_folder, config_dict["results_folder"])
    if gt_folder is not None:
        compute_metrics_and_save_json(
            config_dict, gt_folder, str(Path(results_folder).joinpath("predictions")), prediction_suffix=prediction_suffix
        )


if __name__ == "__main__":
    main()


def run_25d_ensembled_evaluation_for_folder(input_folder, config_dict, output_folder, results_folder):
    model_folders = [str(Path(results_folder).joinpath("fold_{}".format(fold))) for fold in range(config_dict["n_folds"])]

    ensembled_evaluator = HiveEnsembled2Dto3DEvaluator(
        str(Path(output_folder)),
        config_dict,
        pre_processing_25D_transform,
        post_processing_25D_transform,
        model_path_list=model_folders,
    )

    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    val_files = [{"image": filepath} for filepath in subfiles(input_folder, suffix=config_dict["FileExtension"])]
    val_ds = monai.data.Dataset(data=val_files)
    val_loader = monai.data.DataLoader(val_ds, batch_size=1, num_workers=1, collate_fn=list_data_collate)

    ensembled_evaluator.run(val_loader)
