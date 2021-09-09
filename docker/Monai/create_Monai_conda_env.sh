source /opt/conda/bin/activate
conda init bash
conda config --set auto_activate_base true
conda create -n Monai --clone base -y

echo 'conda activate Monai' >> /home/$user/.bashrc

