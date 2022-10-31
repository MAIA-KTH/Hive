#!/usr/bin/env python

import datetime
import os
import json
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from textwrap import dedent

from Hive.utils.log_utils import add_verbosity_options_to_argparser
from Hive.utils.file_utils import subfolders
from Hive.utils.seg_mask_utils import semantic_segmentation_to_instance

TIMESTAMP = "{:%Y-%m-%d_%H-%M-%S}".format(datetime.datetime.now())

DESC = dedent(
    """
    Script to convert a semantic segmentation dataset (with the `Patient ID` as the folder name) into an instance segmentation dataset.
    Instance segmentation masks are saved within the same patient folder with the standard format "INST_SEG.nii.gz". Regions in instance 
    segmentation containing less than 10 voxels are ignored and the number of labels in each instance segmentation mask is saved in a 
    separate json file ('inst_seg_labels.json') alongside its 'Patient ID'. 
    """  # noqa: E501
)
EPILOG = dedent(
    """
    Example call:
    ::
        {filename}  --data-folder /PATH/TO/SEMANTIC_SEG_DATA
    """.format(  # noqa: E501
        filename=Path(__file__).stem
    )
)


def get_arg_parser():
    pars = ArgumentParser(description=DESC, epilog=EPILOG, formatter_class=RawTextHelpFormatter)

    pars.add_argument(
        "--data-folder",
        type=str,
        required=True,
        help="AutoPET patient dataset folder.",
    )

    add_verbosity_options_to_argparser(pars)

    return pars


def main():
    parser = get_arg_parser()
    arguments = vars(parser.parse_args())

    # data_folder = /home/Data/LymphNoDet/AutoPET_Data
    # subject = PETCT_0011f3deaf_0
    subjects = subfolders(arguments["data_folder"], join=False)

    # /home/Data/LymphNoDet/AutoPET_Data/PETCT_0011f3deaf_0/PETCT_0011f3deaf_0_SEG.nii.gz
    # subject + sem_seg = "PETCT_0011f3deaf_0_SEG.nii.gz"
    sem_seg = "_SEG.nii.gz"
    inst_seg = "_INST_SEG.nii.gz"
    labels_dict = {}
    for subject in subjects:
        subject_sem_seg_filename = os.path.join(arguments["data_folder"], subject, str(subject + sem_seg))
        subject_inst_seg_filename = os.path.join(arguments["data_folder"], subject, str(subject + inst_seg))
        num_features = semantic_segmentation_to_instance(subject_sem_seg_filename, subject_inst_seg_filename)
        labels_dict.update({str(subject): num_features})

    # Create Json file with number of labels of instance segmentation for each patient.
    with open('/home/Data/LymphNoDet/inst_seg_labels.json', 'w') as json_file:
        json.dump(labels_dict, json_file)


if __name__ == "__main__":
    main()
