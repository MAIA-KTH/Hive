FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

# SSH-PYCHARM
RUN apt-get update && apt-get install -y openssh-server
RUN apt-get install -y sudo screen git nano tmux vim python3-pip python3-tk
RUN mkdir /var/run/sshd

EXPOSE 22

COPY conda_requirements.txt conda_requirements.txt
COPY start.sh start.sh
COPY .env .env
COPY create_nnUNet_conda_env.sh create_nnUNet_conda_env.sh
RUN git clone https://github.com/MIC-DKFZ/nnUNet.git

RUN pip install -e nnUNet

ENTRYPOINT ["./start.sh"]
CMD ["/usr/sbin/sshd", "-D"]
