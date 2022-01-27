FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

# SSH-PYCHARM
RUN apt-get update && apt-get install -y openssh-server build-essential
RUN apt-get install -y sudo screen git nano tmux vim curl

RUN mkdir /var/run/sshd

EXPOSE 22

COPY start.sh start.sh
COPY docker_env/.env .env
COPY sshd_config /etc/ssh/
COPY create_Hive_conda_env.sh create_Hive_conda_env.sh

ENTRYPOINT ["sh", "./start.sh"]
CMD ["/usr/sbin/sshd", "-D"]
