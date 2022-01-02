#!/usr/bin/env python

import datetime
import json
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from textwrap import dedent

import pandas as pd
import tqdm

from Hive.utils.file_utils import subfolders
from Hive.utils.log_utils import add_verbosity_options_to_argparser, get_logger, log_lvl_from_verbosity_args
from Hive.utils.vector_field_plots import get_vector_field_summary_for_case

TIMESTAMP = "{:%Y-%m-%d_%H-%M-%S}".format(datetime.datetime.now())

DESC = dedent(
    """
    From a given dataset folder, generates a vector field summary for each subject, as well as a dataset summary report,
    including the single summaries for each subject. The vector field summary includes information about the Center of Mass
    for each label class, and the 3D vector components of the vector sum.
    """  # noqa: E501
)
EPILOG = dedent(
    """
    Example call:
    ::
        {filename} --data-folder /PATH/TO/DATA_FOLDER --image-suffix _image.nii.gz --vector-field-suffix _LVC_map.nii.gz --label-suffix _mask.nii.gz
        {filename} --data-folder /PATH/TO/DATA_FOLDER --image-suffix _image.nii.gz --vector-field-suffix _LVC_map.nii.gz --label-suffix _mask.nii.gz --output-suffix _vector_field_summary.xlsx

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
        "--image-suffix",
        type=str,
        required=True,
        help="Image filename suffix to correctly detect the image files in the dataset",
    )

    pars.add_argument(
        "--vector-field-suffix",
        type=str,
        required=True,
        help="Vector field filename suffix to correctly detect the vector field files in the dataset",
    )

    pars.add_argument(
        "--label-suffix",
        type=str,
        required=True,
        default=None,
        help="Label filename suffix to correctly detect the label files in the dataset",
    )

    pars.add_argument(
        "--output-suffix",
        type=str,
        required=False,
        default="_vector_field_summary.xlsx",
        help="Output filename suffix used to save the vector field summary. File extensions can be: '.xlsx', '.csv', '.pkl'. ",
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

    with open(Path(arguments["data_folder"]).joinpath("data_config.json")) as json_file:
        config_dict = json.load(json_file)

    dataset_name = Path(arguments["data_folder"]).stem.replace("_", " ")
    summary = []
    subjects = subfolders(arguments["data_folder"], join=False)
    for subject in tqdm.tqdm(subjects, desc="Vector Field Summary"):
        if (
            Path(arguments["data_folder"]).joinpath(subject, subject + arguments["image_suffix"]).is_file()
            and Path(arguments["data_folder"]).joinpath(subject, subject + arguments["vector_field_suffix"]).is_file()
            and Path(arguments["data_folder"]).joinpath(subject, subject + arguments["label_suffix"]).is_file()
        ):

            output_filename = Path(arguments["data_folder"]).joinpath(subject, subject + arguments["output_suffix"])
            if output_filename.is_file():
                if str(output_filename).endswith(".xlsx"):
                    case_summary_pd = pd.read_excel(output_filename, index_col=0)
                elif str(output_filename).endswith(".csv"):
                    case_summary_pd = pd.read_csv(output_filename, index_col=0)
                elif str(output_filename).endswith(".pkl"):
                    case_summary_pd = pd.read_pickle(output_filename, index_col=0)
                else:
                    raise ValueError("Output file format not recognized, expected one of: '.xslx', '.csv', '.pkl' ")

                n_rows = len(case_summary_pd)
                case_summary_dict = case_summary_pd.to_dict(orient="list")
                case_summary = []
                for row in range(n_rows):
                    row_dict = {}
                    for key in case_summary_dict.keys():
                        row_dict[key] = case_summary_dict[key][row]
                    case_summary.append(row_dict)
            else:
                case_summary = get_vector_field_summary_for_case(
                    str(Path(arguments["data_folder"]).joinpath(subject, subject + arguments["image_suffix"])),
                    str(Path(arguments["data_folder"]).joinpath(subject, subject + arguments["vector_field_suffix"])),
                    str(Path(arguments["data_folder"]).joinpath(subject, subject + arguments["label_suffix"])),
                    subject,
                    config_dict["label_dict"][0],
                )
                if str(output_filename).endswith(".xlsx"):
                    pd.DataFrame(case_summary).to_excel(output_filename)
                elif str(output_filename).endswith(".csv"):
                    pd.DataFrame(case_summary).to_csv(str(output_filename))
                elif str(output_filename).endswith(".pkl"):
                    pd.DataFrame(case_summary).to_pickle(str(output_filename))
                else:
                    raise ValueError("Output file format not recognized, expected one of: '.xslx', '.csv', '.pkl' ")

            _ = [summary.append(row) for row in case_summary]

    if arguments["output_suffix"].endswith(".xlsx"):
        pd.DataFrame(summary).to_excel(Path(arguments["data_folder"]).joinpath(dataset_name + arguments["output_suffix"]))
    elif arguments["output_suffix"].endswith(".csv"):
        pd.DataFrame(summary).to_excel(Path(arguments["data_folder"]).joinpath(dataset_name + arguments["output_suffix"]))
    elif arguments["output_suffix"].endswith(".pkl"):
        pd.DataFrame(summary).to_excel(Path(arguments["data_folder"]).joinpath(dataset_name + arguments["output_suffix"]))
    else:
        raise ValueError("Output file format not recognized, expected one of: '.xslx', '.csv', '.pkl' ")


if __name__ == "__main__":
    main()
