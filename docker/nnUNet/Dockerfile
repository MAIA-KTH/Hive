FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

# SSH-PYCHARM
RUN apt-get update && apt-get install -y openssh-server build-essential
RUN apt-get install -y sudo screen git nano tmux vim curl

RUN mkdir /var/run/sshd

EXPOSE 22

COPY start.sh start.sh
COPY .env .env
COPY create_nnUNet_conda_env.sh create_nnUNet_conda_env.sh

ENTRYPOINT ["./start.sh"]
CMD ["/usr/sbin/sshd", "-D"]
