FROM python:3.8

RUN pip install -U pyyaml
ARG DISABLE_CACHE

WORKDIR /home/jeff/MLOps-yolov5/preprocessing
ENTRYPOINT ["python", "write-yaml.py" ]


# docker build -t tjems6498/kfp-yolov5-preprocess:v0.1 .
# docker push tjems6498/kfp-yolov5-preprocess:v0.1