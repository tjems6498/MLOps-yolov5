# Tesla V100
FROM nvcr.io/nvidia/pytorch:21.10-py3

# RTX A6000, Tesla A100
#FROM nvcr.io/nvidia/pytorch:20.10-py3

# Install linux packages
RUN apt update && apt install -y zip htop screen libgl1-mesa-glx

# code clone

# Install python dependencies
RUN python -m pip install --upgrade pip
RUN pip uninstall -y nvidia-tensorboard nvidia-tensorboard-plugin-dlprof

WORKDIR /home/jeff/MLOps-yolov5/yolov5

RUN pip freeze > requirements.txt coremltools onnx gsutil notebook wandb>=0.12.2
#RUN pip install --no-cache -r coremltools onnx gsutil notebook wandb>=0.12.2
RUN pip install --no-cache -U Pillow
# RUN pip install --no-cache -U torch torchvision numpy
RUN apt-get update

ENTRYPOINT ["python", "train.py"]


# docker build -t tjems6498/kfp-yolov5-train:v0.1 .
# docker push tjems6498/kfp-yolov5-train:v0.1