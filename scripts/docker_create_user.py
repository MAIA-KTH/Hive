#!/usr/bin/env python

import re
import subprocess
from getpass import getpass
from os.path import expanduser
from pathlib import Path
from textwrap import dedent

from Hive.utils.log_utils import str2bool

DESC = dedent(
    """
    Script to create and configure a new User to access Hive in Docker Compose.
    """  # noqa: E501 W291 W605
)
EPILOG = dedent(
    """
    Example call:
        {filename}
    """.format(  # noqa: E501 W291
        filename=Path(__file__).name
    )
)


def generate_and_upload_git_key():
    try:
        subprocess.run("ssh-keygen -t rsa -b 4096 -f {}/.ssh/hive_git_key".format(expanduser("~")))
        subprocess.run("gh auth login")
        subprocess.run('gh ssh-key add {}/.ssh/hive_git_key.pub --title "Hive_git_key"'.format(expanduser("~")))
        subprocess.run("gh auth logout")
    except KeyboardInterrupt:
        return None
    return "{}/.ssh/hive_git_key".format(expanduser("~"))


def main():
    try:
        username = input("Enter new Username:\n")
        password = getpass("Enter password:\n")
        repeat_password = getpass("Repeat password:\n")
        user_email = input("Please Enter a valid e-mail:\n")
        regex = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        if not re.fullmatch(regex, user_email):
            exit(1)
        try:
            add_git_key = input("Do you want to create and upload a Git key? [Y/n]:\n")
            if str2bool(add_git_key):
                git_key = generate_and_upload_git_key()
            else:
                git_key = None
        except KeyboardInterrupt:
            git_key = None
    except KeyboardInterrupt:
        exit(1)

    if password == repeat_password:

        f = open(Path(__file__).parent.parent.joinpath("docker", "Hive", "credentials.env"), "w")
        f.write("user={}\n".format(username))
        f.write("password={}\n".format(password))
        f.write("email={}\n".format(user_email))
        if git_key is not None:
            key_f = open(git_key + ".pub")
            public_key = key_f.read()
            f.write('AUTHORIZED_KEYS="{}"'.format(public_key))
            key_f.close()
            f_env = open(Path(__file__).parent.parent.joinpath("docker", "Hive", ".env"), "w")
            f_env.write("GIT_KEY_PATH={}".format(git_key))
            f_env.close()
        f.close()

    else:
        exit(1)


if __name__ == "__main__":
    main()
