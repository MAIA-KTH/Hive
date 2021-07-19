import os


def test_nnunet_environment():
    results_folder = os.environ["RESULTS_FOLDER"]  # noqa: F841
    preprocessing_folder = os.environ["nnUNet_preprocessed"]  # noqa: F841
    nnunet_base_folder = os.environ["nnUNet_raw_data_base"]  # noqa: F841

    return
