# Tesla V100
FROM nvcr.io/nvidia/pytorch:21.10-py3

# RTX A6000, Tesla A100
#FROM nvcr.io/nvidia/pytorch:20.10-py3

# Install linux packages
RUN apt update && apt install -y zip htop screen libgl1-mesa-glx

# code clone
RUN git clone https://github.com/tjems6498/MLOps-yolov5.git working-dir
WORKDIR working-dir/yolov5

# Install python dependencies
RUN python -m pip install --upgrade pip
RUN pip uninstall -y nvidia-tensorboard nvidia-tensorboard-plugin-dlprof
RUN pip install --no-cache -r requirements.txt coremltools onnx gsutil notebook wandb>=0.12.2
RUN pip install --no-cache -U Pillow
# RUN pip install --no-cache -U torch torchvision numpy
RUN apt-get update

ARG DISABLE_CACHE
# update code
RUN git pull origin master

ENTRYPOINT ["python", "train.py"]


