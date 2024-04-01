import os
import argparse
import torch

from networks.net_factory_3d import net_factory_3d_1,net_factory_3d
from test_3d_patch import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/LA/', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='SSNet', help='exp_name')
parser.add_argument('--model', type=str,  default='vnet', help='model_name')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--detail', type=int,  default=1, help='print metrics for every samples?')
parser.add_argument('--nms', type=int, default=1, help='apply NMS post-procssing?')
parser.add_argument('--labelnum', type=int, default=2, help='labeled data')

FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = "../model/LA/Mean_Teacher_8/vnet_1"
test_save_path = "../model/LA/{}_predictions/".format(FLAGS.exp, FLAGS.labelnum, FLAGS.model)
num_classes = 2

if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)
print(test_save_path)
with open(FLAGS.root_path + 'test.list', 'r') as f:
    image_list = f.readlines()
image_list = [FLAGS.root_path + "/data/" + item.replace('\n', '') + "/mri_norm2.h5" for item in image_list]


def test_calculate_metric():
    model = net_factory_3d(net_type=FLAGS.model, in_chns=1, class_num=num_classes,mode = "test")
    
    save_model_path = os.path.join(snapshot_path, 'vnet_best_model.pth')
    model.load_state_dict(torch.load(save_model_path))
    print("init weight from {}".format(save_model_path))
    model.eval()

    avg_metric = test_all_case(model, image_list, num_classes=num_classes,
                           patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                           save_result=True, test_save_path=test_save_path,
                           metric_detail=FLAGS.detail, nms=FLAGS.nms)

    return avg_metric


if __name__ == '__main__':
    metric = test_calculate_metric()
    print(metric)

# python test_LA.py --model 0214_re01 --gpu 0
