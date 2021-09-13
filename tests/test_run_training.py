import json
import os

import seg_metrics.seg_metrics as sg
from Hive.utils.file_utils import subfiles


def test_training_command():
    config_file = "/home/LungLobeSeg/nnUNet_results/LungLobeSeg_3d_fullres_100_LungLobeSeg_3D_Single_Modality.json"
    fold = 0
    with open(config_file) as json_file:
        data = json.load(json_file)
        unknown_arguments = ["--npz"]
        arguments = [
            data["TRAINING_CONFIGURATION"],
            data["TRAINER_CLASS_NAME"],
            "Task" + data["Task_ID"] + "_" + data["Task_Name"],
            str(fold),
        ]
        arguments.extend(unknown_arguments)
        print("nnUNet_train " + " ".join(arguments))


def test_run_prediction():
    config_file = "/home/LungLobeSeg/nnUNet_results/LungLobeSeg_3d_fullres_100_LungLobeSeg_3D_Single_Modality.json"
    input_folder = "/home/LungLobeSeg/nnUNet_base/nnUNet_raw_data/Task100_LungLobeSeg_3D_Single_Modality/imagesTs"
    output_folder = "/home/LungLobeSeg/nnUNet_results/nnUNet/3d_fullres/Task100_LungLobeSeg_3D_Single_Modality/nnUNetTrainerV2__nnUNetPlansv2.1/predictionsTs"  # noqa: E501
    with open(config_file) as json_file:
        data = json.load(json_file)
        unknown_arguments = ["--save-npz"]
        arguments = [
            "-i",
            input_folder,
            "-o",
            output_folder,
            "-m",
            data["TRAINING_CONFIGURATION"],
            "-t",
            "Task" + data["Task_ID"] + "_" + data["Task_Name"],
        ]
        arguments.extend(unknown_arguments)
        print("nnUNet_predict " + " ".join(arguments))


def test_evaluate_metrics():
    config_file = "/home/LungLobeSeg/nnUNet_results/LungLobeSeg_3d_fullres_100_LungLobeSeg_3D_Single_Modality.json"
    gt_folder = "/home/LungLobeSeg/nnUNet_base/nnUNet_raw_data/Task100_LungLobeSeg_3D_Single_Modality/imagesTs"
    pred_folder = "/home/LungLobeSeg/nnUNet_base/nnUNet_raw_data/Task100_LungLobeSeg_3D_Single_Modality/imagesTs"
    output_folder = "/home/LungLobeSeg/nnUNet_results/nnUNet/3d_fullres/Task100_LungLobeSeg_3D_Single_Modality/nnUNetTrainerV2__nnUNetPlansv2.1/predictionsTs"  # noqa: E501

    with open(config_file) as json_file:
        config_dict = json.load(json_file)

        file_suffix = config_dict["FileExtension"]
        label_dict = config_dict["label_dict"]
        label_dict.pop("0", None)

    labels = label_dict.keys()
    labels = [int(label) for label in labels]

    gt_files = subfiles(gt_folder, join=False, suffix=file_suffix)
    pred_files = subfiles(pred_folder, join=False, suffix=file_suffix)
    for gt_filepath in gt_files:
        if gt_filepath in pred_files:
            csv_filename = os.path.join(output_folder, gt_filepath[: -len(file_suffix)] + ".csv")
            gdth_file = os.path.join(gt_folder, gt_filepath)
            pred_file = os.path.join(pred_folder, gt_filepath)
            sg.write_metrics(
                labels=labels,
                gdth_path=gdth_file,
                pred_path=pred_file,
                csv_file=csv_filename,
            )
        else:
            print("File is Missing")
