mkdir -p /home/$user/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /home/$user/miniconda3/miniconda.sh
bash /home/$user/miniconda3/miniconda.sh -b -u -p /home/$user/miniconda3
rm -rf /home/$user/miniconda3/miniconda.sh

source /home/$user/miniconda3/bin/activate
conda init bash
conda config --set auto_activate_base true
conda install pip
conda create -n nnUNet_to_TF_model_converter --clone base -y
/home/$user/miniconda3/envs/nnUNet_to_TF_model_converter/bin/pip install tensorflow-gpu==2.5.0
/home/$user/miniconda3/envs/nnUNet_to_TF_model_converter/bin/pip install nnunet
/home/$user/miniconda3/envs/nnUNet_to_TF_model_converter/bin/pip install onnx
/home/$user/miniconda3/envs/nnUNet_to_TF_model_converter/bin/pip install onnx-tf

echo 'conda activate nnUNet_to_TF_model_converter' >> /home/$user/.bashrc