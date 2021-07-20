import os
import shutil
from multiprocessing import Pool

import SimpleITK as sitk
from utils.file_utils import subfiles


def test_mp_copy():
    modality = 0
    num_threads = 5
    file_extension = ".nii.gz"
    image_suffix = "_image.nii.gz"
    label_suffix = "_mask.nii.gz"
    image_subpath = "images"
    labels_subpath = "labels"
    input_data_folder = "/home/LungLobeSeg/dataset_147_cases"
    output_data_folder = "/home/test_mp"
    subjects = []
    modality_code = "_{0:04d}".format(modality)

    pool = Pool(num_threads)
    copied_files = []
    for directory in subjects:

        files = subfiles(
            os.path.join(input_data_folder, directory),
            join=False,
            suffix=file_extension,
        )

        image_filename = directory + image_suffix

        if label_suffix is not None:

            label_filename = directory + label_suffix

            if image_filename in files and label_filename in files:
                updated_image_filename = image_filename.replace(image_suffix, modality_code + file_extension)
                updated_label_filename = label_filename.replace(label_suffix, file_extension)
                copied_files.append(
                    pool.starmap_async(
                        copy_image_file,
                        (
                            (
                                os.path.join(input_data_folder, directory, image_filename),
                                os.path.join(output_data_folder, image_subpath, updated_image_filename),
                            ),
                        ),
                    )
                )
                copied_files.append(
                    pool.starmap_async(
                        copy_label_file,
                        (
                            (
                                os.path.join(input_data_folder, directory, directory + image_suffix),
                                os.path.join(input_data_folder, directory, directory + label_suffix),
                                os.path.join(output_data_folder, labels_subpath, updated_label_filename),
                            ),
                        ),
                    )
                )

        else:
            updated_image_filename = image_filename.replace(image_suffix, modality_code + file_extension)
            copied_files.append(
                pool.starmap_async(
                    copy_image_file,
                    (
                        (
                            os.path.join(input_data_folder, directory, image_filename),
                            os.path.join(output_data_folder, image_subpath, updated_image_filename),
                        ),
                    ),
                )
            )
    _ = [i.get() for i in copied_files]


def copy_image_file(input_filepath: str, output_filepath: str):
    shutil.copy(
        input_filepath,
        output_filepath,
    )


def copy_label_file(input_image: str, input_label: str, output_filepath: str):
    label_itk = sitk.ReadImage(input_label)
    image_itk = sitk.ReadImage(input_image)
    label_itk.CopyInformation(image_itk)
    sitk.WriteImage(label_itk, output_filepath)
