#!/bin/bash

echo $user
groupadd --gid 1111 docker_user_group
useradd -m -s /bin/bash -u 1000 --groups docker_user_group $user
echo $user":"$password | chpasswd
adduser $user sudo

cd /home/$user

su $user "/workspace/create_nnUNet_conda_env.sh"

ln -s /home/user/LungLobeSeg /home/$user/LungLobeSeg
ln -s /home/user/nnunet_data_preparation /home/$user/nnunet_data_preparation
exec "$@"
