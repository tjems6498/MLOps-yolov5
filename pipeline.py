import kfp
from kfp import dsl
from kfp import onprem
# TODO: dataloader num_workers 값을 주었을때 shm 문제해결

def preprocess_op(pvc_name, volume_name, volume_mount_path):

    return dsl.ContainerOp(
        name='Preprocess Data',
        image='tjems6498/kfp-yolov5-preprocess:v0.1',
        arguments=['--data-path', volume_mount_path + '/data',
                   '--label-path', volume_mount_path + '/data/classname.txt'],
    ).apply(onprem.mount_pvc(pvc_name, volume_name=volume_name, volume_mount_path=volume_mount_path))

def train_op(pvc_name, volume_name, volume_mount_path, img, batch, epochs, weights):

    return dsl.ContainerOp(
        name='train Data',
        image='tjems6498/kfp-yolov5-train:v0.3',
        arguments=['--img', img,
                   '--batch', batch,
                   '--epochs', epochs,
                   '--data', volume_mount_path + '/MLOps-yolov5/yolov5/data/custom.yaml',
                   '--weights', weights],
    ).apply(onprem.mount_pvc(pvc_name, volume_name=volume_name, volume_mount_path=volume_mount_path)).set_gpu_limit(4)



@dsl.pipeline(
    name='Yolov5 Pipeline',
    description=''
)
def surface_pipeline(pretrain: bool,
                     image_size: int,
                     batch_size: int,
                     epochs: int,
                     weights_path: str
                     ):
    pvc_name = "workspace-yolov5"
    volume_name = 'pipeline'
    volume_mount_path = '/home/jeff'

    with dsl.Condition(pretrain == True):
        _preprocess_op = preprocess_op(pvc_name, volume_name, volume_mount_path)

    _train_op = train_op(pvc_name, volume_name, volume_mount_path, image_size, batch_size, epochs, weights_path).after(_preprocess_op)


if __name__ == '__main__':
    kfp.compiler.Compiler().compile(surface_pipeline, './kfp.yaml')
