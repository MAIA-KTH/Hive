source /opt/conda/bin/activate
conda init bash
conda config --set auto_activate_base true
conda create -n nnUNet --clone base
conda activate nnUNet
conda install -c conda-forge --file /workspace/conda_requirements.txt -y
conda env config vars set $(eval echo $(cat /workspace/.env))
