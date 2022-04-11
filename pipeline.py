import kfp
from kfp import dsl
from kfp import onprem
# TODO: dataloader num_workers 값을 주었을때 shm 문제해결

def preprocess_op(pvc_name, volume_name, volume_mount_path):

    return dsl.ContainerOp(
        name='Preprocess Data',
        image='tjems6498/kfp-yolov5-preprocess:v0.1',
        arguments=['--data-path', volume_mount_path,
                   '--data-path', volume_mount_path + '/classname.txt'],
    ).apply(onprem.mount_pvc(pvc_name, volume_name=volume_name, volume_mount_path=volume_mount_path))


@dsl.pipeline(
    name='Surface Crack Pipeline',
    description=''
)
def surface_pipeline():
    pvc_name = "workspace-yolov5"
    volume_name = 'pipeline'
    volume_mount_path = '/home/jeff'


    _preprocess_op = preprocess_op(pvc_name, volume_name, volume_mount_path)


if __name__ == '__main__':
    kfp.compiler.Compiler().compile(surface_pipeline, './kfp.yaml')
