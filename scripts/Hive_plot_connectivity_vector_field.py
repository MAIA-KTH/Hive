#!/usr/bin/env python

import datetime
import json
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from textwrap import dedent

import pandas as pd

from tqdm import tqdm
from Hive.utils.file_utils import subfolders
from Hive.utils.log_utils import add_verbosity_options_to_argparser, get_logger, log_lvl_from_verbosity_args, str2bool
from Hive.utils.vector_field_plots import (
    create_2D_GIF_for_vector_field,
    create_plotly_3D_vector_field,
    create_plotly_3D_vector_field_summary,
    get_sphericaL_coordinates_df,
    plot_spherical_coordinates,
)

TIMESTAMP = "{:%Y-%m-%d_%H-%M-%S}".format(datetime.datetime.now())

DESC = dedent(
    """
    Generates vector field [ LVC ] dataset summary plots, given the LVC dataset summary file.
    """  # noqa: E501
)
EPILOG = dedent(
    """
    Example call:
    ::
        {filename} --data-folder /path/to/data_folder --image-suffix _image.nii.gz --vector-field-suffix _LVC_map.nii.gz
        {filename} --data-folder /path/to/data_folder --image-suffix _image.nii.gz --vector-field-suffix _LVC_map.nii.gz --plot-2D y --plot-3D n --orientation coronal
        {filename} --data-folder /path/to/data_folder --image-suffix _image.nii.gz --vector-field-suffix _LVC_map.nii.gz --vector-field-summary-suffix _LVC_summary.xlsx
        {filename} --data-folder /path/to/data_folder --image-suffix _image.nii.gz --vector-field-suffix _LVC_map.nii.gz --label-suffix _mask.nii.gz --output-suffix _LVC_field
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
        "--plot-2D",
        type=str2bool,
        required=False,
        default="n",
        help="Flag to save 2D GIFs for each subject. Default: No",
    )

    pars.add_argument(
        "--plot-3D",
        type=str2bool,
        required=False,
        default="y",
        help="Flag to save 3D plots for each subject and the 3D spherical summary plot. Default: Yes",
    )

    pars.add_argument(
        "--orientation",
        type=str,
        required=False,
        default="axial",
        help="Orientation for 2D slicing when creating 2D GIFs.",
    )

    pars.add_argument(
        "--image-suffix",
        type=str,
        required=True,
        help="Filepath suffix to load the Image volume for each Subject.",
    )

    pars.add_argument(
        "--vector-field-suffix",
        type=str,
        required=True,
        help="Filepath suffix to load the LVC field volume for each Subject.",
    )

    pars.add_argument(
        "--vector-field-summary-suffix",
        type=str,
        required=False,
        default="_vector_field_summary.xlsx",
        help="Filepath suffix used to save the Vector Field Summary file and the 3D spherical summary plot (as HTML).",
    )

    pars.add_argument(
        "--label-suffix",
        type=str,
        required=False,
        default=None,
        help="Optional Filepath suffix, used to load the Lobe mask file and save the 2D GIFs, grouping the vector sum per lobe.",
    )

    pars.add_argument(
        "--output-suffix",
        type=str,
        required=False,
        default="_vector_field",
        help="Filepath suffix to save the optional GIF files and the Subject 3D plots (as HTML).",
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
    if Path(arguments["data_folder"]).joinpath(dataset_name + arguments["vector_field_summary_suffix"]).is_file():
        spherical_df = get_sphericaL_coordinates_df(
            str(Path(arguments["data_folder"]).joinpath(dataset_name + arguments["vector_field_summary_suffix"])),
            labels,
        )
        spherical_df_output_path = Path(arguments["data_folder"]).joinpath(
            dataset_name + "_spherical" + arguments["vector_field_summary_suffix"]
        )
        if str(spherical_df_output_path).endswith(".xlsx"):
            pd.DataFrame(spherical_df).to_excel(spherical_df_output_path)
        elif str(spherical_df_output_path).endswith(".csv"):
            pd.DataFrame(spherical_df).to_csv(str(spherical_df_output_path))
        elif str(spherical_df_output_path).endswith(".pkl"):
            pd.DataFrame(spherical_df).to_pickle(str(spherical_df_output_path))
        else:
            raise ValueError("Output file format not recognized, expected one of: '.xslx', '.csv', '.pkl' ")
        manova_df = plot_spherical_coordinates(
            spherical_df,
            str(
                Path(arguments["data_folder"]).joinpath(
                    dataset_name + Path(arguments["vector_field_summary_suffix"]).stem + "_spherical.html"
                )
            ),
            dataset_name,
            config_dict["label_dict"][0],
        )

        spherical_manova_df_output_path = Path(arguments["data_folder"]).joinpath(
            dataset_name + "_spherical_manova" + arguments["vector_field_summary_suffix"]
        )
        if str(spherical_manova_df_output_path).endswith(".xlsx"):
            pd.DataFrame(manova_df).to_excel(spherical_manova_df_output_path)
        elif str(spherical_manova_df_output_path).endswith(".csv"):
            pd.DataFrame(manova_df).to_csv(str(spherical_manova_df_output_path))
        elif str(spherical_manova_df_output_path).endswith(".pkl"):
            pd.DataFrame(manova_df).to_pickle(str(spherical_manova_df_output_path))
        else:
            raise ValueError("Output file format not recognized, expected one of: '.xslx', '.csv', '.pkl' ")

        if arguments["plot_3D"]:
            create_plotly_3D_vector_field_summary(
                str(Path(arguments["data_folder"]).joinpath(dataset_name + arguments["vector_field_summary_suffix"])),
                Path(arguments["data_folder"]).joinpath(
                    dataset_name + Path(arguments["vector_field_summary_suffix"]).stem + ".html"
                ),
                dataset_name,
            )

    subjects = subfolders(arguments["data_folder"], join=False)
    for subject in tqdm(subjects):
        if (
            Path(arguments["data_folder"]).joinpath(subject, subject + arguments["image_suffix"]).is_file()
            and Path(arguments["data_folder"]).joinpath(subject, subject + arguments["vector_field_suffix"]).is_file()
        ):
            if arguments["plot_2D"]:
                output_filename = Path(arguments["data_folder"]).joinpath(
                    subject, subject + "_" + arguments["orientation"] + arguments["output_suffix"] + ".gif"
                )
                if (
                    arguments["label_suffix"] is None
                    or not Path(arguments["data_folder"]).joinpath(subject, subject + arguments["label_suffix"]).is_file()
                ):
                    create_2D_GIF_for_vector_field(
                        str(Path(arguments["data_folder"]).joinpath(subject, subject + arguments["image_suffix"])),
                        str(Path(arguments["data_folder"]).joinpath(subject, subject + arguments["vector_field_suffix"])),
                        arguments["orientation"],
                        output_filename,
                    )

                else:
                    create_2D_GIF_for_vector_field(
                        str(Path(arguments["data_folder"]).joinpath(subject, subject + arguments["image_suffix"])),
                        str(Path(arguments["data_folder"]).joinpath(subject, subject + arguments["vector_field_suffix"])),
                        arguments["orientation"],
                        output_filename,
                        mask_filename=str(Path(arguments["data_folder"]).joinpath(subject, subject + arguments["label_suffix"])),
                    )
            if arguments["plot_3D"]:
                output_filename = Path(arguments["data_folder"]).joinpath(subject, subject + arguments["output_suffix"] + ".html")
                create_plotly_3D_vector_field(
                    subject,
                    str(Path(arguments["data_folder"]).joinpath(subject, subject + arguments["image_suffix"])),
                    str(Path(arguments["data_folder"]).joinpath(subject, subject + arguments["vector_field_summary_suffix"])),
                    output_filename,
                    config_dict["label_dict"][0],
                )


if __name__ == "__main__":
    main()
