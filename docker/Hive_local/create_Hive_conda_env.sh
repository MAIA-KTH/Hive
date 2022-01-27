source /opt/conda/bin/activate
conda init bash
conda config --set auto_activate_base true
conda create -n Hive --clone base -y

echo 'conda activate Hive' >> /home/$user/.bashrc