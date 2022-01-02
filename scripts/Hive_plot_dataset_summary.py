#!/usr/bin/env python

import datetime
import json
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from textwrap import dedent

import pandas as pd

from Hive.utils.dataset_plots import get_dataset_plots
from Hive.utils.log_utils import add_verbosity_options_to_argparser, get_logger, log_lvl_from_verbosity_args

TIMESTAMP = "{:%Y-%m-%d_%H-%M-%S}".format(datetime.datetime.now())

DESC = dedent(
    """
    Generates dataset summary plots, given the dataset summary file.
    """  # noqa: E501
)
EPILOG = dedent(
    """
    Example call:
    ::
        {filename} --data-folder /path/to/data_folder
        {filename} --data-folder /path/to/data_folder --summary-file-suffix _summary.csv

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
        help="Dataset folder.",
    )

    pars.add_argument(
        "--summary-file-suffix",
        type=str,
        required=False,
        default="_summary.xlsx",
        help="Filepath used to save the dataset summary file.",
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
    dataset_name = Path(arguments["data_folder"]).stem.replace("_", " ")
    with open(Path(arguments["data_folder"]).joinpath("data_config.json")) as json_file:
        config_dict = json.load(json_file)
    labels = []
    for label_idx in config_dict["label_dict"][0]:
        if label_idx != "0":
            labels.append(config_dict["label_dict"][0][label_idx])
    summary_file_path = Path(arguments["data_folder"]).joinpath(dataset_name + arguments["summary_file_suffix"])
    if summary_file_path.is_file():
        if arguments["summary_file_suffix"].endswith(".xlsx"):
            summary_pd = pd.read_excel(summary_file_path, index_col=0)
        elif arguments["summary_file_suffix"].endswith(".csv"):
            summary_pd = pd.read_csv(summary_file_path, index_col=0)
        elif arguments["summary_file_suffix"].endswith(".pkl"):
            summary_pd = pd.read_pickle(summary_file_path, index_col=0)
        else:
            raise ValueError("Output file format not recognized, expected one of: '.xslx', '.csv', '.pkl' ")
        get_dataset_plots(
            summary_pd,
            str(Path(arguments["data_folder"]).joinpath(dataset_name + Path(arguments["summary_file_suffix"]).stem)),
            labels,
            dataset_name=dataset_name,
        )


if __name__ == "__main__":
    main()
