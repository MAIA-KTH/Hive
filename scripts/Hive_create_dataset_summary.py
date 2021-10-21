#!/usr/bin/env python

import datetime
import json
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from textwrap import dedent

import pandas as pd

from Hive.utils.file_utils import subfolders
from Hive.utils.log_utils import add_verbosity_options_to_argparser, get_logger, log_lvl_from_verbosity_args
from Hive.utils.volume_utils import compute_subject_summary

TIMESTAMP = "{:%Y-%m-%d_%H-%M-%S}".format(datetime.datetime.now())

DESC = dedent(
    """
    For a specified data folder, creates a JSON summary file, containing volumetric and geometric information for each volume.
    The 3D image volume to be considered is specified with **image_modality**, matching the corresponding suffix in the
    configuration file. The label suffix is automatically detected from the configuration file.
    """  # noqa: E501
)
EPILOG = dedent(
    """
    Example call:
    ::
        {filename} --data-folder /path/to/data_folder --config-file LungLobeSeg_2.5D_config.json  --image-modality CT --summary-csv-file /path/to/summary.json
    """.format(  # noqa: E501
        filename=Path(__file__).name
    )
)


def get_arg_parser():
    pars = ArgumentParser(description=DESC, epilog=EPILOG, formatter_class=RawTextHelpFormatter)

    pars.add_argument(
        "--data-folder",
        type=str,
        required=True,
        help="Dataset folder",
    )

    pars.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="Configuration JSON file with experiment and dataset parameters ",
    )

    pars.add_argument(
        "--image-modality",
        type=str,
        required=True,
        help="Image modality considered as the 3D volume image. Example: ``CT`` ",
    )

    pars.add_argument(
        "--summary-csv-file",
        type=str,
        required=True,
        help="Filepath where to save JSON summary file.",
    )

    add_verbosity_options_to_argparser(pars)

    return pars


def main():
    parser = get_arg_parser()
    arguments = vars(parser.parse_args())

    logger = get_logger(  # NOQA: F841
        name=Path(__file__).name,
        level=log_lvl_from_verbosity_args(arguments),
    )

    with open(arguments["config_file"]) as json_file:
        config_dict = json.load(json_file)

    image_suffix = config_dict["Modalities"][arguments["image_modality"]]
    subjects = subfolders(arguments["data_folder"], join=False)
    df_summary = pd.DataFrame()
    for subject in subjects:
        subject_dict = {
            "image": str(Path(arguments["data_folder"]).joinpath(subject, subject + image_suffix)),
            "label": str(Path(arguments["data_folder"]).joinpath(subject, subject + config_dict["label_suffix"])),
        }
        subject_summary = compute_subject_summary(subject_dict, config_dict["label_dict"])
        df_summary = df_summary.append(subject_summary, ignore_index=True)

    df_summary.to_csv(arguments["summary_csv_file"])


if __name__ == "__main__":
    main()
