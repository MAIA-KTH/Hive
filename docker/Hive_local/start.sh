#!/bin/bash

useradd -m -s /bin/bash -u 1000 $user
echo $user":"$password | chpasswd
adduser $user sudo

su $user "/workspace/create_Hive_conda_env.sh"

echo 'source /workspace/.env' >> /home/$user/.bashrc
echo 'export receiver_email='$email >> /home/$user/.bashrc

exec "$@"
