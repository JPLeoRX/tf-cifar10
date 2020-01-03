FROM tensorflow/tensorflow:1.14.0-gpu-py3
RUN apt update
RUN apt install -y git
RUN pip install tensorflow-datasets