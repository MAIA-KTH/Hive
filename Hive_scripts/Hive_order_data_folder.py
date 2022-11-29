#!/usr/bin/env python

import os
import sys
from argparse import ArgumentParser, RawTextHelpFormatter
from textwrap import dedent

from Hive.utils.file_utils import order_data_in_single_folder, order_data_folder_by_patient
from Hive.utils.log_utils import (
    get_logger,
    log_lvl_from_verbosity_args,
    add_verbosity_options_to_argparser,
    str2bool,
)

DESC = dedent(
    """
    Order Dataset folder creating and moving the files in the corresponding patient subdirectories
    Folder structure :
        <output_dir>/<patient_id>/<dataset_patient_files>
    The suffix used to find the patient IDs is set via the --patient-suffix flag:
        Filename -> [patient_ID + patient_suffix]
    Example:
        --patient-suffix _image.nii.gz
        PatientA_image.nii.gz  -> [patient_ID = PatientA]
    If -in-place [yes/no] is set to 'yes', --output-folder is not considered and the folder ordering is performed on the input folder
    """  # noqa: E501
)
EPILOG = dedent(
    """
    Example call:
    ::
        {filename} -i /path/to/inputfolder -o /path/to/outputfolder --patient-suffix _image.nii.gz
        {filename} -i /path/to/inputfolder -o /path/to/outputfolder --patient-suffix _image.nii.gz
        {filename} -i /path/to/inputfolder --patient-suffix _image.nii.gz --in-place yes
    """.format(  # noqa: E501
        filename=os.path.basename(__file__)
    )
)


def get_arg_parser():
    parser = ArgumentParser(description=DESC, epilog=EPILOG, formatter_class=RawTextHelpFormatter)

    parser.add_argument(
        "-i",
        "--input-folder",
        type=str,
        required=True,
        help="Input Dataset folder",
    )

    parser.add_argument(
        "-o",
        "--output-folder",
        type=str,
        required=not ("--in-place" in sys.argv and str2bool(sys.argv[sys.argv.index("--in-place") + 1])),
        help="Output folder where to save the ordered Dataset",
    )

    parser.add_argument(
        "--patient-suffix",
        type=str,
        required=True,
        help="Suffix used to find the patient IDs from the filenames",
    )

    parser.add_argument(
        "--file-extension",
        type=str,
        default=".nii.gz",
        required=False,
        help="File extension for the files in the dataset. (Default: `.nii.gz`)",
    )

    parser.add_argument(
        "--in-place",
        type=str2bool,
        required=False,
        default="no",
        help='if set to "yes", the output folder matches the input folder',
    )

    parser.add_argument(
        "--create-subject-subfolders",
        type=str2bool,
        required=False,
        default="yes",
        help='If set to "yes", individual subject subfolders are created, with the corresponding files stored. If set '
             'to "no", all the files are directly saved in the main data folder.',
    )

    parser.add_argument(
        "--assign-parent-dir-name",
        type=str2bool,
        required=False,
        default="no",
        help='If set to "yes", the parent directory name for all the files will be appended to the filename. To be used when'
             'the files for the same patient are saved in different folders ("image", "mask"), with the same name',
    )

    add_verbosity_options_to_argparser(parser)

    return parser


def main():
    parser = get_arg_parser()
    args = vars(parser.parse_args())

    logger = get_logger(  # noqa: F841
        name=os.path.basename(__file__),
        level=log_lvl_from_verbosity_args(args),
    )

    if args["in_place"]:
        args["output_folder"] = args["input_folder"]

    if args['assign_parent_dir_name']:
        order_data_in_single_folder(args["input_folder"], args["output_folder"], True,
                                    file_extension=args['file_extension'])
    else:
        order_data_in_single_folder(args["input_folder"], args["output_folder"])
    if args["create_subject_subfolders"]:
        order_data_folder_by_patient(args["output_folder"], args["patient_suffix"])


if __name__ == "__main__":
    main()
