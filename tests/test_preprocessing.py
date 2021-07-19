import json


def test_preprocessing_command():
    config_file = "/home/LungLobeSeg/nnUNet_results/LungLobeSeg_3d_fullres_100_LungLobeSeg_3D_Single_Modality.json"

    with open(config_file) as json_file:
        data = json.load(json_file)
        unknown_arguments = ["--verify_dataset_integrity"]
        arguments = [
            "-t",
            data["Task_ID"],
        ]
        arguments.extend(unknown_arguments)
        print("nnUNet_plan_and_preprocess " + " ".join(arguments))
