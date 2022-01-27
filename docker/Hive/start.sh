#!/bin/bash

useradd -m -s /bin/bash -u 1000 $user
echo $user":"$password | chpasswd
adduser $user sudo

su $user "/workspace/create_Hive_conda_env.sh"

echo 'source /workspace/.env' >> /home/$user/.bashrc
echo 'export receiver_email='$email >> /home/$user/.bashrc
echo 'alias Hive_install="pip install git+ssh://git@github.com/SimoneBendazzoli93/Hive.git"' >> /home/$user/.bashrc
#pip install git+https://github.com/SimoneBendazzoli93/Hive.git@v1.1

if [ "${AUTHORIZED_KEYS}" != "**None**" ]; then
    echo "=> Found authorized keys"

    mkdir -p /home/simone/.ssh
    chown simone /home/simone/.ssh
    chmod 700 /home/simone/.ssh
    touch /home/simone/.ssh/authorized_keys
    chown simone /home/simone/.ssh/authorized_keys
    chmod 600 /home/simone/.ssh/authorized_keys

    IFS=$','

    arr=$(echo ${AUTHORIZED_KEYS})

    for x in $arr
    do
        x=$(echo $x | sed -e 's/^ *//' -e 's/ *$//')
        cat /home/simone/.ssh/authorized_keys | grep "$x" >/dev/null 2>&1
        if [ $? -ne 0 ]; then
            echo "=> Adding public key to .ssh/authorized_keys: $x"
            echo "$x" >> /home/simone/.ssh/authorized_keys
        fi
    done
else
    echo "ERROR: No authorized keys found in \$AUTHORIZED_KEYS"
    exit 1
fi
exec "$@"
