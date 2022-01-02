#! /usr/bin/env python

import shutil
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from textwrap import dedent

from Hive.utils.file_utils import subfolders
from Hive.utils.log_utils import get_logger, add_verbosity_options_to_argparser, log_lvl_from_verbosity_args, str2bool
from Hive.utils.mialab_utils import run_mialab_lung_lobe_annotation_command
from Hive.utils.volume_utils import combine_annotations_and_generate_label_map

DESC = dedent(
    """
    Tool to annotate Lung Lobe dataset, creating the lobe map in MiaLab
    """  # noqa: E501
)
EPILOG = dedent(
    """
     {filename} --data-folder /path/to/data_folder --mialab-folder /path/to/mialab_folder --image-suffix _image.nii.gz --label-suffix _mask.nii.gz
     {filename} --data-folder /path/to/data_folder --mialab-folder /path/to/mialab_folder --image-suffix _image.nii.gz --label-suffix _mask.nii.gz --skip-existing y
    """.format(  # noqa: E501
        filename=Path(__file__).name
    )
)
label_dict = {"base": "_lung_map.nii.gz", "2": "_L.nii.gz", "4": "_RU.nii.gz", "5": "_RL.nii.gz"}


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
        "--skip-existing",
        type=str2bool,
        required=False,
        default=True,
        help="Specify to rerun the process for already existing output files.",
    )

    pars.add_argument(
        "--label-suffix",
        type=str,
        required=True,
        help="Label filename suffix to save the label files in the dataset",
    )

    pars.add_argument(
        "--image-suffix",
        type=str,
        required=True,
        help="Image filename suffix to save the image files in the dataset",
    )

    pars.add_argument(
        "--file-extension",
        type=str,
        required=False,
        default=".nii.gz",
        help="",
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

    subjects = subfolders(arguments["data_folder"], join=False)
    for subject in subjects:
        if arguments["skip_existing"]:
            if not Path(arguments["data_folder"]).joinpath(subject, subject + arguments["label_suffix"]).is_file():

                run_mialab_lung_lobe_annotation_command(
                    arguments["mialab_folder"],
                    arguments["data_folder"],
                    subject,
                    arguments["file_extension"],
                )
                combine_annotations_and_generate_label_map(
                    subject, arguments["data_folder"], label_dict, arguments["label_suffix"]
                )
                shutil.move(
                    Path(arguments["data_folder"]).joinpath(subject, subject + arguments["file_extension"]),
                    Path(arguments["data_folder"]).joinpath(subject, subject + arguments["image_suffix"]),
                )
        else:
            run_mialab_lung_lobe_annotation_command(
                arguments["mialab_folder"],
                arguments["data_folder"],
                subject,
                arguments["file_extension"],
            )
            combine_annotations_and_generate_label_map(subject, arguments["data_folder"], label_dict, arguments["label_suffix"])
            shutil.move(
                Path(arguments["data_folder"]).joinpath(subject, subject + arguments["file_extension"]),
                Path(arguments["data_folder"]).joinpath(subject, subject + arguments["image_suffix"]),
            )


if __name__ == "__main__":
    main()
