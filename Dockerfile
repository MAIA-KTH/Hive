FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

ARG username='USERNAME'
ARG user_pw='USER_PASSWORD'

ENV nnUNet_raw_data_base="/home/simone/Data/LungLobeSeg/nnUNet_dataset"
ENV nnUNet_preprocessed="/home/simone/Data/LungLobeSeg/nnUNet_preprocessing"
ENV RESULTS_FOLDER="/home/simone/Data/LungLobeSeg/nnUNet_results"
# SSH-PYCHARM
RUN apt-get update && apt-get install -y openssh-server
RUN apt-get install -y sudo screen git nano tmux vim python3-pip
RUN mkdir /var/run/sshd
RUN groupadd --gid 1111 docker_user_group \
  && useradd -m -s /bin/bash -u 1001 --groups docker_user_group $username \
  && echo $username":"$user_pw | chpasswd \
  && adduser $username sudo

EXPOSE 22

RUN chown -R $username:$username /home/$username

WORKDIR /home/$username

COPY conda_requirements.txt conda_requirements.txt
RUN git clone https://github.com/MIC-DKFZ/nnUNet.git
WORKDIR nnUNet
RUN pip install -e .



CMD ["/usr/sbin/sshd", "-D"]
