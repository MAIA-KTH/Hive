import os
import datetime

from argparse import ArgumentParser, RawTextHelpFormatter
from textwrap import dedent



TIMESTAMP = "{:%Y-%m-%d_%H-%M-%S}".format(datetime.datetime.now())

DESC = dedent(
    """

    """  # noqa: E501
)
EPILOG = dedent(
    """
    
    """.format(  # noqa: E501
        filename=os.path.basename(__file__)
    )
)


if __name__ == "__main__":
    parser = ArgumentParser(
        description=DESC, epilog=EPILOG, formatter_class=RawTextHelpFormatter
    )

    parser.add_argument(
        "-i",
        "--input-data-folder",
        type=str,
        required=True,
        help="Input Dataset folder",
    )

    parser.add_argument(
        "-o"
        "--nnUNet-raw-data-base",
        type=str,
        required=True,
        help="nnUNet_raw_data_base folder path",
    )

    parser.add_argument(
        "--task-ID",
        type=int,
        default="100",
        help="Task ID used in the folder path tree creation",
    )

    parser.add_argument(
        "--task-name",
        type=str,
        default="LungLobeSeg_3D_Single_Modality",
        help="Task Name used in the folder path tree creation",
    )

    parser.add_argument(
        "--image-suffix",
        type=str,
        required=True,
        help="Image filename suffix to correctly detect the image files in the dataset",
    )

    parser.add_argument(
        "--label-suffix",
        type=str,
        required=True,
        help="Label filename suffix to correctly detect the label files in the dataset",
    )

    args = vars(parser.parse_args())
