#!/usr/bin/env python

import datetime
import json
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from textwrap import dedent

from Hive.utils.file_utils import match_subject_IDs_by_suffix_length
from Hive.utils.log_utils import add_verbosity_options_to_argparser, get_logger, log_lvl_from_verbosity_args
from Hive.utils.volume_utils import compare_lung_volumes

TIMESTAMP = "{:%Y-%m-%d_%H-%M-%S}".format(datetime.datetime.now())

DESC = dedent(
    """
    Match and group the Dataset volumes, according to the given Minimum Filename Pattern Length that they must share and
    according to the number of volumes that should be included in the single groups. For each formed group, the volumes
    are assigned to the "Inhalation" category or to the "Exhalation" category, according to the volume size.
    A JSON summary in then saved, including the information for each case and to which category it is assigned.
    """  # noqa: E501
)
EPILOG = dedent(
    """
    Example call:
    ::
        {filename} --data-folder /PATH/TO/DATA_FOLDER --image-suffix _image.nii.gz --label-suffix _mask.nii.gz
        {filename} --data-folder /PATH/TO/DATA_FOLDER --image-suffix _image.nii.gz --label-suffix _mask.nii.gz --group-size 2 --matching-pattern-length 8
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
        help="Dataset folder ",
    )

    pars.add_argument(
        "--image-suffix",
        type=str,
        required=True,
        help="Image filename suffix to correctly detect the image files in the dataset",
    )

    pars.add_argument(
        "--label-suffix",
        type=str,
        required=True,
        help="Label filename suffix to correctly detect the image files in the dataset",
    )

    pars.add_argument(
        "--group-size",
        type=int,
        required=False,
        default=2,
        help="Number of files that should be included in each group. If the group size is different, all the included"
        "volumes are not assigned to a category. (Default: 2)",
    )

    pars.add_argument(
        "--matching-pattern-length",
        type=int,
        required=False,
        default=10,
        help="Length of the pattern (starting from the beginning of the filename ) to consider when matching the files."
        " (Default: 10)",
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
    group_size = arguments["group_size"]
    data_folder = arguments["data_folder"]
    image_suffix = arguments["image_suffix"]
    label_suffix = arguments["label_suffix"]
    matching_pattern_length = arguments["matching_pattern_length"]

    subject_id_phases = {}
    matching_IDs = match_subject_IDs_by_suffix_length(data_folder, matching_pattern_length)
    for matching_ID in matching_IDs:
        if len(matching_ID) != group_size:
            for id in matching_ID:
                subject_id_phases[id] = "Not Assigned"
        else:
            filename_dict_list = [
                {
                    "image": str(Path(data_folder).joinpath(ID, ID + image_suffix)),
                    "label": str(Path(data_folder).joinpath(ID, ID + label_suffix)),
                }
                for ID in matching_ID
            ]
            volume_size_in_L = compare_lung_volumes(filename_dict_list)
            indexes = list(range(len(matching_ID)))
            min_volume_idx = volume_size_in_L.index(min(volume_size_in_L))
            indexes.remove(min_volume_idx)
            subject_id_phases[matching_ID[min_volume_idx]] = "Exhalation"
            for idx in indexes:
                subject_id_phases[matching_ID[idx]] = "Inhalation"

    with open(Path(data_folder).joinpath("breathing_phase_summary.json"), "w") as fp:
        json.dump(subject_id_phases, fp)


if __name__ == "__main__":
    main()
