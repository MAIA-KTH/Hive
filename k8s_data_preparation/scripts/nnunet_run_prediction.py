import json
import os
from argparse import ArgumentParser, RawTextHelpFormatter
from textwrap import dedent

from utils.log_utils import (
    get_logger,
    add_verbosity_options_to_argparser,
    log_lvl_from_verbosity_args,
)

DESC = dedent(
    """
    Run nnUNet command to run inference and save the predictions for a set of volumes.
    """  # noqa: E501
)
EPILOG = dedent(
    """
    Example call:
     {filename} -i /INPUT_FOLDER -o /OUTPUT_FOLDER --config-file ../LungLobeSeg_3d_fullres_100_LungLobeSeg_3D_Single_Modality.json
     {filename} -i /INPUT_FOLDER -o /OUTPUT_FOLDER --config-file ../LungLobeSeg_3d_fullres_100_LungLobeSeg_3D_Single_Modality.json --save_npz
    """.format(  # noqa: E501
        filename=os.path.basename(__file__)
    )
)

if __name__ == "__main__":
    parser = ArgumentParser(description=DESC, epilog=EPILOG, formatter_class=RawTextHelpFormatter)

    parser.add_argument(
        "-i",
        "--input-folder",
        type=str,
        required=True,
        help="Folder path containing the volumes to be predicted",
    )

    parser.add_argument(
        "-o",
        "--output-folder",
        type=str,
        required=True,
        help="Folder path where to save the predictions",
    )

    parser.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="File path for the configuration dictionary, used to retrieve experiments variables (Task_ID)",
    )

    add_verbosity_options_to_argparser(parser)

    arguments, unknown_arguments = parser.parse_known_args()
    args = vars(arguments)

    logger = get_logger(
        name=os.path.basename(__file__),
        level=log_lvl_from_verbosity_args(args),
    )

    config_file = args["config_file"]

    with open(config_file) as json_file:
        data = json.load(json_file)

        arguments = [
            "-i",
            args["input_folder"],
            "-o",
            args["output_folder"],
            "-m",
            data["TRAINING_CONFIGURATION"],
            "-t",
            "Task" + data["Task_ID"] + "_" + data["Task_Name"],
        ]
        arguments.extend(unknown_arguments)
        os.system("nnUNet_predict " + " ".join(arguments))
