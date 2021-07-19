import json
import os
from argparse import ArgumentParser, RawTextHelpFormatter
from textwrap import dedent

import seg_metrics.seg_metrics as sg
from utils.file_utils import subfiles
from utils.log_utils import get_logger, add_verbosity_options_to_argparser, log_lvl_from_verbosity_args, DEBUG

DESC = dedent(
    """
    Run segmentation metrics evaluation for a set of volumes and store the results in a CSV file.

    """  # noqa: E501
)
EPILOG = dedent(
    """
    {filename} --ground-truth-folder /GT_FOLDER --prediction-folder /PRED_FOLDER --output-folder /OUT_FOLDER --config-file ../LungLobeSeg_3d_fullres_100_LungLobeSeg_3D_Single_Modality.json
    """.format(  # noqa: E501
        filename=os.path.basename(__file__)
    )
)

if __name__ == "__main__":
    parser = ArgumentParser(description=DESC, epilog=EPILOG, formatter_class=RawTextHelpFormatter)

    parser.add_argument(
        "--ground-truth-folder",
        type=str,
        required=True,
        help="Folder path containing the Ground Truth volumes",
    )

    parser.add_argument(
        "--prediction-folder",
        type=str,
        required=True,
        help="Folder path containing the prediction volumes",
    )

    parser.add_argument(
        "--output-folder",
        type=str,
        required=True,
        help="Folder path where to save the CSV metrics table",
    )

    parser.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="File path for the configuration dictionary, used to retrieve experiments variables ",
    )

    add_verbosity_options_to_argparser(parser)
    args = vars(parser.parse_args())

    logger = get_logger(
        name=os.path.basename(__file__),
        level=log_lvl_from_verbosity_args(args),
    )

    with open(args["config_file"]) as json_file:
        config_dict = json.load(json_file)

        file_suffix = config_dict["FileExtension"]
        label_dict = config_dict["label_dict"]
        label_dict.pop("0", None)

    labels = label_dict.keys()
    labels = [int(label) for label in labels]

    gt_files = subfiles(args["ground_truth_folder"], join=False, suffix=file_suffix)
    pred_files = subfiles(args["prediction_folder"], join=False, suffix=file_suffix)
    for gt_filepath in gt_files:
        if gt_filepath in pred_files:
            csv_filename = os.path.join(args["output_folder"], gt_filepath[: -len(file_suffix)] + ".csv")
            gdth_file = os.path.join(args["ground_truth_folder"], gt_filepath)
            pred_file = os.path.join(args["prediction_folder"], gt_filepath)
            logger.log(DEBUG, " Predicting {}".format(pred_file))
            metrics = sg.write_metrics(labels=labels, gdth_path=gdth_file, pred_path=pred_file, csv_file=csv_filename)
        else:
            logger.log(DEBUG, " File {} is missing".format(gt_filepath))
