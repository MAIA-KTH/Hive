#! /usr/bin/env python

import json
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from textwrap import dedent

from Hive.utils.file_utils import subfolders
from Hive.utils.log_utils import get_logger, add_verbosity_options_to_argparser, log_lvl_from_verbosity_args, str2bool
from Hive.utils.mialab_utils import run_mialab_fuzzy_segmentation_command
from Hive.utils.volume_utils import erode_mask

DESC = dedent(
    """
    Creates a Lung Vessel Connectivity map from a Fuzzy Segmentation, computed in MiaLab.
    """  # noqa: E501
)
EPILOG = dedent(
    """
    Example call:
    ::
        {filename} --data-folder /path/to/data_folder --image-suffix _image.nii.gz --label-suffix _mask.nii.gz --output-suffix _connectivity_map.nii.gz
        {filename} --data-folder /path/to/data_folder --image-suffix _image.nii.gz --label-suffix _mask.nii.gz --output-suffix _connectivity_map.nii.gz --phase-json-file /path/to/json/phase_file.json
        {filename} --data-folder /path/to/data_folder --image-suffix _image.nii.gz --label-suffix _mask.nii.gz --output-suffix _connectivity_map.nii.gz --phase-json-file /path/to/json/phase_file.json --phases Exhalation
    """.format(  # noqa: E501
        filename=Path(__file__).name
    )
)


def get_arg_parser():
    pars = ArgumentParser(description=DESC, epilog=EPILOG, formatter_class=RawTextHelpFormatter)

    pars.add_argument(
        "-i",
        "--data-folder",
        type=str,
        required=True,
        help="Dataset folder",
    )

    pars.add_argument(
        "--mialab-folder",
        type=str,
        required=True,
        help="Folder path for MiaLab executable",
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
        help="Label filename suffix to correctly detect the label files in the dataset",
    )

    pars.add_argument(
        "--output-suffix",
        type=str,
        required=True,
        help="Output filename suffix used to save the Lung Vessel Connectivity map",
    )

    pars.add_argument(
        "--phase-json-file",
        type=str,
        required=False,
        help="JSON file used to map each subject with the corresponding breathing phase.",
    )

    pars.add_argument(
        "--phases",
        type=str,
        nargs="+",
        required=False,
        help="If specified, only subjects in the listed phases are processed.",
    )

    pars.add_argument(
        "--skip-existing",
        type=str2bool,
        required=False,
        default=True,
        help="Specify to rerun the process for already existing output files.",
    )

    pars.add_argument(
        "--erosion-iterations",
        type=int,
        required=False,
        default=0,
        help="Specify to number of iterations to optionally erode the mask image.",
    )
    add_verbosity_options_to_argparser(pars)

    return pars


def main():
    parser = get_arg_parser()

    arguments = vars(parser.parse_args())

    logger = get_logger(  # noqa: F841
        name=Path(__file__).name,
        level=log_lvl_from_verbosity_args(arguments),
    )
    phase_dict = {}
    phases = []
    if arguments["phase_json_file"] is not None:
        with open(arguments["phase_json_file"]) as json_file:
            phase_dict = json.load(json_file)
            phases = list(set(phase_dict.values()))

    if arguments["phases"] is not None:
        phases = arguments["phases"]

    subjects = subfolders(arguments["data_folder"], join=False)
    for subject in subjects:
        if arguments["skip_existing"]:
            if Path(arguments["data_folder"]).joinpath(subject, subject + arguments["output_suffix"]).is_file():
                continue
        if len(phase_dict.keys()) > 0 and phase_dict[subject] in phases:
            label_suffix = arguments["label_suffix"]
            if arguments["erosion_iterations"] > 0:
                label_suffix = "_eroded_mask.nii.gz"
                label_filename = str(Path(arguments["data_folder"]).joinpath(subject, subject + arguments["label_suffix"]))
                eroded_label_filename = str(Path(arguments["data_folder"]).joinpath(subject, subject + label_suffix))
                erode_mask({"label": label_filename}, arguments["erosion_iterations"], eroded_label_filename)

            run_mialab_fuzzy_segmentation_command(
                arguments["mialab_folder"],
                arguments["data_folder"],
                subject,
                arguments["image_suffix"],
                arguments["output_suffix"],
                label_suffix,
            )
        elif len(phase_dict.keys()) == 0:
            label_suffix = arguments["label_suffix"]
            if arguments["erosion_iterations"] > 0:
                label_suffix = "_eroded_mask.nii.gz"
                label_filename = str(Path(arguments["data_folder"]).joinpath(subject, subject + arguments["label_suffix"]))
                eroded_label_filename = str(Path(arguments["data_folder"]).joinpath(subject, subject + label_suffix))
                erode_mask({"label": label_filename}, arguments["erosion_iterations"], eroded_label_filename)

            run_mialab_fuzzy_segmentation_command(
                arguments["mialab_folder"],
                arguments["data_folder"],
                subject,
                arguments["image_suffix"],
                arguments["output_suffix"],
                label_suffix,
            )


if __name__ == "__main__":
    main()
