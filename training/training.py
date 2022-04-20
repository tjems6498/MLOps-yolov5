import os
from yolov5.train import parse_opt, main
from mlflow.tracking.client import MlflowClient
from mlflow.pytorch import save_model
from yolov5.models.common import DetectMultiBackend

# Model
def upload_model_to_mlflow(opt):
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio-service.kubeflow.svc:9000"
    os.environ["AWS_ACCESS_KEY_ID"] = "minio"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
    client = MlflowClient("http://mlflow-server-service.mlflow-system.svc:5000")

    model = DetectMultiBackend(f'runs/train/{opt.name}/weights/best.pt', device=opt.device)

    conda_env = {'name': 'mlflow-env', 'channels': ['conda-forge'],
                 'dependencies': ['python=3.9.4', 'pip', {'pip': ['mlflow', 'torch==1.8.0', 'cloudpickle==2.0.0']}]}

    save_model(
        pytorch_model=model,
        path=opt.name,
        conda_env=conda_env,
    )

    tags = {"DeepLearning": "yolov5 plastic detection"}
    run = client.create_run(experiment_id="1", tags=tags)
    client.log_artifact(run.info.run_id, opt.model_name)


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)

    upload_model_to_mlflow(opt)