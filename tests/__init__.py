from dotenv import dotenv_values
import os


def load_env(dot_env_file):
    env_dict = dotenv_values(dot_env_file)
    for env_var in env_dict:
        os.environ[env_var] = env_dict[env_var]


dot_env_file_path = "../docker/nnUNet/.env"
load_env(dot_env_file_path)
