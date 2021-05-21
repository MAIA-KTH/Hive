import shutil
import os
import SimpleITK as sitk

def create_nnunet_data_folder_tree(data_folder: str, task_name: str, task_id: str):
    """
    Create nnUNet_raw_data_base folder tree, ready to be populated with the dataset

    :param data_folder: folder path corresponding to the nnUNet_raw_data_base ENV variable
    :param task_id: string used as task_id when creating task folder
    :param task_name: string used as task_name when creating task folder
    """
    os.makedirs(os.path.join(data_folder, 'nnUNet_raw_data', 'Task'+task_id+'_'+task_name, 'imagesTr'), exist_ok=True)
    os.makedirs(os.path.join(data_folder, 'nnUNet_raw_data', 'Task' + task_id + '_' + task_name, 'labelsTr'),
                exist_ok=True)


def copy_images_to_nnunet_data_folder(input_data_folder: str, nnunet_data_folder: str, image_suffix: str):
    """

    :param input_data_folder:
    :param nnunet_data_folder:
    :param image_suffix:
    """
    for root, dirs, _ in os.walk(input_data_folder):
        for directory in dirs:
            for _, _, files in os.walk(os.path.join(input_data_folder, directory)):
                for file in files:
                    if file == (directory + image_suffix):
                        nnunet_image_filename = file.replace(image_suffix, '.nii.gz')
                        image_1=sitk.ReadImage(os.path.join(root,directory,directory+image_suffix))
                        image_2 = sitk.ReadImage(os.path.join(root, directory, directory +'_image.nii.gz'))
                        image_1.CopyInformation(image_2)
                        sitk.WriteImage(image_1,os.path.join(nnunet_data_folder, nnunet_image_filename))
                        #shutil.copy(os.path.join(root, directory, file), os.path.join(nnunet_data_folder, nnunet_image_filename))
