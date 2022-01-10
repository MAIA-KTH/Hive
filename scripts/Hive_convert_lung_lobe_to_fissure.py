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
from Hive.utils.volume_utils import convert_lung_label_map_to_fissure_mask

TIMESTAMP = "{:%Y-%m-%d_%H-%M-%S}".format(datetime.datetime.now())

DESC = dedent(
    """
    Convert a dataset of Lung Lobe label maps into Lung Fissures binary masks.
    """  # noqa: E501
)
EPILOG = dedent(
    """
    Example call:
    ::
        {filename} --data-folder /PATH/TO/DATA_FOLDER --label-suffix _mask.nii.gz --fissure-suffix _fissure_mask.nii.gz
        {filename} --data-folder /PATH/TO/DATA_FOLDER --label-suffix _mask.nii.gz --fissure-suffix _fissure_mask.nii.gz --dilation-iterations 2 --n-workers 5
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
        help="Dataset folder.",
    )

    pars.add_argument(
        "--label-suffix",
        type=str,
        required=True,
        help="Filename suffix to identify the Lung Lobe label map. ",
    )

    pars.add_argument(
        "--fissure-suffix",
        type=str,
        required=True,
        help="Filename suffix to save the fissure binary mask. ",
    )

    pars.add_argument(
        "--dilation-iterations",
        type=int,
        required=False,
        default=1,
        help="Number of dilation iterations to perform. Default: 1",
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
    pool = Pool(arguments["n_workers"])
    lung_lobe_conversions = []
    for subject in subjects:
        label_filename = str(Path(arguments["data_folder"]).joinpath(subject, subject + arguments["label_suffix"]))
        fissure_filename = str(Path(arguments["data_folder"]).joinpath(subject, subject + arguments["fissure_suffix"]))
        lung_lobe_conversions.append(
            pool.starmap_async(
                convert_lung_label_map_to_fissure_mask,
                ((label_filename, fissure_filename, arguments["dilation_iterations"]),),
            )
        )

    _ = [i.get() for i in tqdm(lung_lobe_conversions)]


if __name__ == "__main__":
    main()
