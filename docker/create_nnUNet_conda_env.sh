source /opt/conda/bin/activate
conda init bash
conda config --set auto_activate_base true
conda create -n nnUNet --clone base
conda activate nnUNet
pip install -r /workspace/conda_requirements.txt
conda env config vars set $(eval echo $(cat /workspace/.env))
conda env config vars set GIT_SSH_COMMAND="ssh -v -i /etc/secret-volume/ssh-privatekey"
pip install git+https://github.com/SimoneBendazzoli93/Hive.git@v1.0
#pip install git+ssh://git@github.com/SimoneBendazzoli93/Hive.git@v1.0