FROM tensorflow/tensorflow:2.5.0-gpu

# SSH-PYCHARM
RUN apt-get update && apt-get install -y openssh-server build-essential
RUN apt-get install -y sudo screen git nano tmux vim curl

RUN mkdir /var/run/sshd

EXPOSE 22

COPY start.sh start.sh

COPY create_nnUNet_to_tf_conda_env.sh /create_nnUNet_to_tf_conda_env.sh
COPY convert_nnunet_model_to_tf.py /convert_nnunet_model_to_tf.py

ENTRYPOINT ["sh", "./start.sh"]
CMD ["/usr/sbin/sshd", "-D"]
