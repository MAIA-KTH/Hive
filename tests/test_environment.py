import os
from . import dot_env_file_path
from dotenv import dotenv_values


def test_nnunet_environment():

    env_dict = dotenv_values(dot_env_file_path)
    results_folder = os.environ["RESULTS_FOLDER"]
    preprocessing_folder = os.environ["nnUNet_preprocessed"]
    nnunet_base_folder = os.environ["nnUNet_raw_data_base"]
    n_threads = os.environ["N_THREADS"]
    assert results_folder == env_dict["RESULTS_FOLDER"]
    assert preprocessing_folder == env_dict["nnUNet_preprocessed"]
    assert nnunet_base_folder == env_dict["nnUNet_raw_data_base"]
    assert n_threads == env_dict["N_THREADS"]
    return
