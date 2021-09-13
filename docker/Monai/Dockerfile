FROM projectmonai/monai:0.6.0

# SSH-PYCHARM
RUN apt-get update && apt-get install -y openssh-server build-essential
RUN apt-get install -y sudo screen git nano tmux vim curl

RUN mkdir /var/run/sshd

EXPOSE 22

WORKDIR /workspace
COPY start.sh start.sh
COPY .env .env
COPY create_Monai_conda_env.sh create_Monai_conda_env.sh

ENTRYPOINT ["./start.sh"]
CMD ["/usr/sbin/sshd", "-D"]