#!/usr/bin/env python

import datetime
import json
import random
import shutil
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from textwrap import dedent

from Hive.utils.log_utils import add_verbosity_options_to_argparser

TIMESTAMP = "{:%Y-%m-%d_%H-%M-%S}".format(datetime.datetime.now())

DESC = dedent(
    """
    Generates and saves a subset, given a dataset. The subset data are extracted from the original dataset according to the
    provided ``classes``. A JSON file mapping each subject to the corresponding class is needed ( ``data_class_file``).
    An optional parameter ``max_size`` can be specified to limit the size of the subset.
    """  # noqa: E501
)
EPILOG = dedent(
    """
    {filename} --data-folder /PATH/TO/DATASET --output-folder /PATH/TO/SUBSET --data-class-file /PATH/TO/SUBJECT_CLASSES.json --subclasses CLASS_1
    {filename} --data-folder /PATH/TO/DATASET --output-folder /PATH/TO/SUBSET --data-class-file /PATH/TO/SUBJECT_CLASSES.json --subclasses CLASS_1 --max-size 100
    """.format(  # noqa: E501
        filename=Path(__file__).stem
    )
)


def main():
    parser = get_arg_parser()

    arguments = vars(parser.parse_args())

    with open(arguments["data_class_file"], "r") as fp:
        data_class_dict = json.load(fp)

    count = 0

    classes = arguments["subclasses"]
    max_size = arguments["max_size"]
    Path(arguments["output_folder"]).mkdir(parents=True, exist_ok=True)
    if max_size is None:
        max_size = len(data_class_dict)

    patients = list(data_class_dict.keys())

    random.shuffle(patients)

    for patient in patients:
        if data_class_dict[patient] in classes and count <= int(max_size):
            if Path(arguments["output_folder"]).joinpath(patient).is_dir():
                ...
            else:
                shutil.copytree(
                    Path(arguments["data_folder"]).joinpath(patient), Path(arguments["output_folder"]).joinpath(patient)
                )
            count += 1


def get_arg_parser():
    pars = ArgumentParser(description=DESC, formatter_class=RawTextHelpFormatter)

    pars.add_argument(
        "--data-folder",
        type=str,
        required=True,
        help="Input dataset folder",
    )

    pars.add_argument(
        "--output-folder",
        type=str,
        required=True,
        help="Output subset folder",
    )

    pars.add_argument(
        "--data-class-file",
        type=str,
        required=True,
        help="JSON file including the class label for each volume in the dataset.",
    )

    pars.add_argument(
        "--subclasses",
        type=str,
        nargs="+",
        required=True,
        help="List of classes from where to select the subset data.",
    )

    pars.add_argument(
        "--max-size",
        type=str,
        required=False,
        help="Maximum size of the generated subset. Default ``None```: no size limit is set.",
    )

    add_verbosity_options_to_argparser(pars)

    return pars


if __name__ == "__main__":
    main()
