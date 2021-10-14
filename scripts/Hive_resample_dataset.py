#!/usr/bin/env python
import datetime
import sys
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from textwrap import dedent

import SimpleITK as sitk

from Hive.utils.file_utils import subfolders
from Hive.utils.log_utils import add_verbosity_options_to_argparser, get_logger, log_lvl_from_verbosity_args, DEBUG
from Hive.utils.volume_utils import resample_image

TIMESTAMP = "{:%Y-%m-%d_%H-%M-%S}".format(datetime.datetime.now())

DESC = dedent(
    """
    Resample a dataset according to the given specifications.
    Three options are available:
    -   Spacing: resampling performed according to the given spacing values. Negative values are used to keep the original spacing.
    -   Size: resampling performed according to the given size values. Negative values are used to keep the original size.
    -   Scale: resampling performed according to the given scale values.
    Accepted interpolation methods are "nn" (Nearest Neighbor) or "linear".
    """  # noqa: E501
)
EPILOG = dedent(
    """
    Example call:
    ::
        {filename} --data-folder /PATH/TO/DATA_FOLDER --input-suffix _image.nii.gz --output-suffix _image_512x.nii.gz  -output-size 512 -1 -1
        {filename} --data-folder /PATH/TO/DATA_FOLDER --input-suffix _image.nii.gz --output-suffix _image_5x.nii.gz  -output-spacing 5 -1 -1  --interpolation-method linear
        {filename} --data-folder /PATH/TO/DATA_FOLDER --input-suffix _image.nii.gz --output-suffix _image_3x.nii.gz  -output-scale 3 1 1
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
        "--input-suffix",
        type=str,
        required=True,
        help="Image filename suffix to correctly detect the image files in the dataset to resample.",
    )

    pars.add_argument(
        "--output-suffix",
        type=str,
        required=True,
        help="Output filename for the resampled image.",
    )

    pars.add_argument(
        "--interpolation-method",
        type=str,
        required=False,
        default="nn",
        help="Interpolation method used in resampling. Defaults: `nn`. Accepted values: ´nn´, ´linear´",
    )

    pars.add_argument(
        "--output-spacing",
        type=float,
        nargs="+",
        required=not ("--output-size" in sys.argv or "--output-scale" in sys.argv),
        default=None,
        help="Spacing values used to resample the images. Negative values are used to keep the original spacing.",
    )

    pars.add_argument(
        "--output-size",
        type=int,
        nargs="+",
        required=not ("--output-scale" in sys.argv or "--output-spacing" in sys.argv),
        default=None,
        help="Size values used to resample the images. Negative values are used to keep the original spacing.",
    )

    pars.add_argument(
        "--output-scale",
        type=float,
        nargs="+",
        default=None,
        required=not ("--output-size" in sys.argv or "--output-spacing" in sys.argv),
        help="Scale values used to resample the images.",
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

    output_resolution = arguments["output_spacing"]
    output_shape = arguments["output_size"]
    interpolation = arguments["interpolation_method"]
    output_scale = arguments["output_scale"]

    subjects = subfolders(arguments["data_folder"], join=False)
    for subject in subjects:
        if output_resolution is not None:
            output_spacing = output_resolution.copy()
        else:
            output_spacing = output_resolution

        if output_shape is not None:
            output_size = output_shape.copy()
        else:
            output_size = output_shape
        input_filename = str(Path(arguments["data_folder"]).joinpath(subject, subject + arguments["input_suffix"]))
        output_filename = str(Path(arguments["data_folder"]).joinpath(subject, subject + arguments["output_suffix"]))
        itk_image = sitk.ReadImage(input_filename)
        initial_spacing, initial_size = itk_image.GetSpacing(), itk_image.GetSize()

        if output_resolution is not None:
            for idx, output_spacing_component in enumerate(output_spacing):
                if output_spacing_component < 0:
                    output_spacing[idx] = initial_spacing[idx]

            scale = [init_spacing / out_spacing for init_spacing, out_spacing in zip(initial_spacing, output_spacing)]
            output_size = [int(init_size * scale_factor) for init_size, scale_factor in zip(initial_size, scale)]

        if output_shape is not None:
            for idx, output_size_component in enumerate(output_size):
                if output_size_component < 0:
                    output_size[idx] = initial_size[idx]

            scale = [out_size / init_size for init_size, out_size in zip(initial_size, output_size)]
            output_spacing = [init_spacing / scale_factor for init_spacing, scale_factor in zip(initial_spacing, scale)]

        if output_scale is not None:
            scale = output_scale
            output_spacing = [init_spacing / scale_factor for init_spacing, scale_factor in zip(initial_spacing, scale)]
            output_size = [int(init_size * scale_factor) for init_size, scale_factor in zip(initial_size, scale)]

        if interpolation == "nn":
            itk_interpolation = sitk.sitkNearestNeighbor
        elif interpolation == "linear":
            itk_interpolation = sitk.sitkLinear
        else:
            raise ValueError("Interpolation method not valid!")

        logger.log(
            DEBUG,
            "Resampling case {}: {} -> {}, {} -> {}".format(subject, initial_size, output_size, initial_spacing, output_spacing),
        )
        itk_image = resample_image(itk_image, output_size, output_spacing, itk_interpolation)
        sitk.WriteImage(itk_image, output_filename)


if __name__ == "__main__":
    main()
