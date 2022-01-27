# Set Up nnUnet_to_TF environment in Docker
1. Set Credentials
    Set username and password in [credentials.env](credentials.env)
2. Build Docker Image

``docker build . -t nnunet_tf``
3. Set optional volume mounts in [docker-compose.yaml](docker-compose.yaml) under:
```
services:
    nnunet:
        volumes:
        - PATH_TO_LOCAL_FOLDER:PATH_TO_CONTAINER_FOLDER
```
4. Set optional values for port mapping and resource requests (GPU, memory, CPUs)
5. Run Docker Compose:
``docker-compose up -d``
6. Access container via SSH:
``ssh username@localhost -p SSH_PORT``

# nnUNet trained model to TensorFlow bundle conversion

Run:

``/convert_nnunet_model_to_tf.py --plans-pickle-file /PATH/TO/PLANS_PICKLE_FILE --model-checkpoint /PATH/TO/CHECKPOINT_FILE --output-folder /PATH/TO/TF_BUNDLE``