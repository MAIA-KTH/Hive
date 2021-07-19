import datetime
import json
import os
from argparse import ArgumentParser, RawTextHelpFormatter
from textwrap import dedent

from utils.file_utils import (
    create_nnunet_data_folder_tree,
    split_dataset,
    copy_data_to_dataset_folder,
    save_config_json,
    generate_dataset_json,
)
from utils.log_utils import (
    get_logger,
    add_verbosity_options_to_argparser,
    log_lvl_from_verbosity_args,
)

TIMESTAMP = "{:%Y-%m-%d_%H-%M-%S}".format(datetime.datetime.now())

DESC = dedent(
    """
    Standardize dataset for 3D LungLobeSeg experiment, to be compatible with nnUNet framework.
    The dataset is assumed to be in NIFTI format (*.nii.gz) and containing only a single
    modality ( CT ) for the input 3D volumes.

    """  # noqa: E501
)
EPILOG = dedent(
    """
     Example call:
      {filename} --input /path/to/input_data_folder --image-suffix _image.nii.gz --label-suffix _mask.nii.gz
      {filename} --input /path/to/input_data_folder --image-suffix _image.nii.gz --label-suffix _mask.nii.gz --task-ID 106 --task-name LungLobeSeg3D
      {filename} --input /path/to/input_data_folder --image-suffix _image.nii.gz --label-suffix _mask.nii.gz --task-ID 101 --task-name 3D_LungLobeSeg --test-split 30 --config-file ./configs/LungLobeSeg_nnUNet_3D_config.json
    """.format(  # noqa: E501
        filename=os.path.basename(__file__)
    )
)


def main(arguments):
    try:
        dataset_path = os.path.join(
            os.environ["nnUNet_raw_data_base"],
            "nnUNet_raw_data",
            "Task" + arguments["task_ID"] + "_" + arguments["task_name"],
        )

    except KeyError:
        logger.error("nnUNet_raw_data_base is not set as environment variable")
        return 1

    with open(arguments["config_file"]) as json_file:
        config_dict = json.load(json_file)

    create_nnunet_data_folder_tree(
        os.environ["nnUNet_raw_data_base"],
        arguments["task_name"],
        arguments["task_ID"],
    )
    train_dataset, test_dataset = split_dataset(
        arguments["input_data_folder"], arguments["test_split"], config_dict["Seed"]
    )
    copy_data_to_dataset_folder(
        arguments["input_data_folder"],
        train_dataset,
        dataset_path,
        arguments["image_suffix"],
        "imagesTr",
        config_dict,
        arguments["label_suffix"],
        "labelsTr",
        0,
    )
    copy_data_to_dataset_folder(
        arguments["input_data_folder"],
        test_dataset,
        dataset_path,
        arguments["image_suffix"],
        "imagesTs",
        config_dict,
        arguments["label_suffix"],
        "labelsTs",
        0,
    )
    generate_dataset_json(
        os.path.join(dataset_path, "dataset.json"),
        os.path.join(dataset_path, "imagesTr"),
        os.path.join(dataset_path, "imagesTs"),
        config_dict["Modalities"],
        config_dict["label_dict"],
        config_dict["DatasetName"],
    )

    config_dict["Task_ID"] = arguments["task_ID"]
    config_dict["Task_Name"] = arguments["task_name"]
    config_dict["base_folder"] = os.environ["nnUNet_raw_data_base"]

    output_json_basename = (
        config_dict["DatasetName"]
        + "_"
        + config_dict["TRAINING_CONFIGURATION"]
        + "_"
        + config_dict["Task_ID"]
        + "_"
        + config_dict["Task_Name"]
        + ".json"
    )

    try:
        config_dict["results_folder"] = os.environ["RESULTS_FOLDER"]
        os.makedirs(
            config_dict["results_folder"],
            exist_ok=True,
        )
    except KeyError:
        logger.warning(
            "RESULTS_FOLDER is not set as environment variable, {} is not saved".format(
                output_json_basename
            )
        )
        return 1
    try:
        config_dict["preprocessing_folder"] = os.environ["nnUNet_preprocessed"]
        os.makedirs(
            config_dict["preprocessing_folder"],
            exist_ok=True,
        )

    except KeyError:
        logger.warning(
            "nnUNet_preprocessed is not set as environment variable, not saved in {}".format(  # noqa E501
                output_json_basename
            )
        )
    save_config_json(
        config_dict, os.path.join(config_dict["results_folder"], output_json_basename)
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
        "--task-ID",
        type=str,
        default="100",
        help="Task ID used in the folder path tree creation (Default: 100)",
    )

    parser.add_argument(
        "--task-name",
        type=str,
        default="LungLobeSeg_3D_Single_Modality",
        help="Task Name used in the folder path tree creation (Default: LungLobeSeg_3D_Single_Modality)",  # noqa E501
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

    parser.add_argument(
        "--test-split",
        type=int,
        choices=range(0, 101),
        metavar="[0-100]",
        default=20,
        help="Split value ( in %% ) to create Test set from Dataset (Default: 20)",
    )

    parser.add_argument(
        "--config-file",
        type=str,
        required=False,
        default="./configs/LungLobeSeg_nnUNet_3D_config.json",
        help="Configuration JSON file with experiment and dataset parameters "
        "(Default: ./configs/LungLobeSeg_nnUNet_3D_config.json)",
    )

    add_verbosity_options_to_argparser(parser)

    args = vars(parser.parse_args())

    logger = get_logger(
        name=os.path.basename(__file__),
        level=log_lvl_from_verbosity_args(args),
    )

    main(args)
