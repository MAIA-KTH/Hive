#!/usr/bin/env python

import datetime
import os
from argparse import ArgumentParser, RawTextHelpFormatter
from multiprocessing import Pool
from pathlib import Path
from textwrap import dedent

from tqdm import tqdm

from Hive.utils.file_utils import subfolders
from Hive.utils.log_utils import add_verbosity_options_to_argparser, get_logger, log_lvl_from_verbosity_args
from Hive.utils.volume_utils import convert_DICOM_folder_to_NIFTI_image

TIMESTAMP = "{:%Y-%m-%d_%H-%M-%S}".format(datetime.datetime.now())

DESC = dedent(
    """
    Script to convert a ``DICOM`` dataset (structured as Patient-Study-Series) into a NIFTI format (with the `Patient ID` as the folder name).
    When multiple studies for the same patient are found, different **DICOM studies** are saved in different folders, appending the study index to the patient name.
    *DICOM series* for the same study are saved in the same patient folder.
    """  # noqa: E501
)
EPILOG = dedent(
    """
    Example call:
    ::
        {filename}  --data-folder /PATH/TO/DICOM_DATA --output-folder /PATH/TO/NIFTI_DATASET
    """.format(  # noqa: E501
        filename=Path(__file__).name
    )
)

if "N_THREADS" not in os.environ:
    os.environ["N_THREADS"] = "1"


def get_arg_parser():
    pars = ArgumentParser(description=DESC, epilog=EPILOG, formatter_class=RawTextHelpFormatter)

    pars.add_argument(
        "--data-folder",
        type=str,
        required=True,
        help="DICOM Dataset folder.",
    )

    pars.add_argument(
        "--output-folder",
        type=str,
        required=True,
        help="Output folder where to save the converted NIFTI dataset.",
    )

    pars.add_argument(
        "--n-workers",
        type=int,
        required=False,
        default=os.environ["N_THREADS"],
        help="Number of worker threads to use. (Default: {})".format(os.environ["N_THREADS"]),
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
    subjects = subfolders(arguments["data_folder"], join=False)

    pool = Pool(int(arguments["n_workers"]))
    Path(arguments["output_folder"]).mkdir(parents=True, exist_ok=True)
    DICOM_to_NIFTI_conversions = []
    for subject in subjects:
        subject_dicom_folder = str(Path(arguments["data_folder"]).joinpath(subject))
        subject_nifti_filename = str(Path(arguments["output_folder"]).joinpath(subject))
        DICOM_to_NIFTI_conversions.append(
            pool.starmap_async(
                convert_DICOM_folder_to_NIFTI_image,
                ((subject_dicom_folder, subject_nifti_filename),),
            )
        )

    _ = [i.get() for i in tqdm(DICOM_to_NIFTI_conversions)]


if __name__ == "__main__":
    main()
