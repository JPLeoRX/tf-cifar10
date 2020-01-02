FROM tensorflow/tensorflow:1.14.0-py3
RUN apt update
RUN apt install -y git
RUN git clone https://github.com/JPLeoRX/tf-cifar10.git
RUN pip install tensorflow-datasets
CMD ["python", "/tf-cifar10/cifar.py"]