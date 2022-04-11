import argparse
import yaml

def write_yamlFile(data_path, label_path):

    with open(label_path, 'r') as f:
        labels = f.read().split()

    yaml_dict = {}
    yaml_dict['train'] = f'{data_path}/train/images'
    yaml_dict['val'] = f'{data_path}/val/images'
    yaml_dict['test'] = f'{data_path}/test/images'
    yaml_dict['nc'] = len(labels)
    yaml_dict['names'] = labels

    with open('../yolov5/data/custom.yaml', 'w') as f:  # yaml 파일은 yolov5/data 폴더에 있어야 함
        yaml.dump(yaml_dict, f)


    print(yaml_dict)

    while 1:
        True



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, help='dataset root path')
    parser.add_argument('--label-path', type=str, help='className txt file')
    opt = parser.parse_args()

    write_yamlFile(opt.data_path, opt.label_path)




