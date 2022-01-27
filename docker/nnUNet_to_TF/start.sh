#!/bin/bash

useradd -m -s /bin/bash -u 1000 $user
echo $user":"$password | chpasswd
adduser $user sudo

su $user "/create_nnUNet_to_tf_conda_env.sh"

exec "$@"
