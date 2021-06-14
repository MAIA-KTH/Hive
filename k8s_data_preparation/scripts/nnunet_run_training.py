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


    """  # noqa: E501
)
EPILOG = dedent(
    """
    {filename}
    """.format(  # noqa: E501
        filename=os.path.basename(__file__)
    )
)

if __name__ == "__main__":
    parser = ArgumentParser(
        description=DESC, epilog=EPILOG, formatter_class=RawTextHelpFormatter
    )

    parser.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="???",
    )

    parser.add_argument(
        "--run-fold",
        type=int,
        default=0,
        help="???",
    )

    add_verbosity_options_to_argparser(parser)

    args = vars(parser.parse_args())

    logger = get_logger(
        name=os.path.basename(__file__),
        level=log_lvl_from_verbosity_args(args),
    )

    config_file = os.path.join(
        os.environ["RESULTS_FOLDER"],
        "LungLobeSeg_3d_fullres_100_LungLobeSeg_3D_Single_Modality.json",
    )

    with open(config_file) as json_file:
        data = json.load(json_file)

        arguments = [
            data["TRAINING_CONFIGURATION"],
            data["TRAINER_CLASS_NAME"],
            "Task" + data["Task_ID"] + "_" + data["Task_Name"],
        ]
        print("nnUNet_train " + " ".join(arguments))
