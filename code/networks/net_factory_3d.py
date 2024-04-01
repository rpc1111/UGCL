from networks.unet_3D import unet_3D
from networks.vnet import VNet
from networks.vnet_bcp import VNet_bcp
from networks.vnet_ssnet import VNet_ssnet
from networks.vnet_mcnet import MCNet3d_v2
from networks.vnet_DTC import VNet_dtc
from networks.VoxResNet import VoxResNet
from networks.attention_unet import Attention_UNet
from networks.nnunet import initialize_network


def net_factory_3d_1(net_type="unet_3D", in_chns=1, class_num=2):
    if net_type == "unet_3D":
        net = unet_3D(n_classes=class_num, in_channels=in_chns).cuda()
    elif net_type == "attention_unet":
        net = Attention_UNet(n_classes=class_num, in_channels=in_chns).cuda()
    elif net_type == "voxresnet":
        net = VoxResNet(in_chns=in_chns, feature_chns=64,
                        class_num=class_num).cuda()
    elif net_type == "vnet":
        net = VNet(n_channels=in_chns, n_classes=class_num,
                   normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "nnUNet":
        net = initialize_network(num_classes=class_num).cuda()
    else:
        net = None
    return net
def net_factory_3d(net_type="unet", in_chns=1, class_num=2, mode = "train"):

    if net_type == "vnet" and mode == "train":
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "vnet" and mode == "test":
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    elif net_type == "vnet_bcp" and mode == "test":
        net=VNet_bcp(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    elif net_type == "vnet_ssnet" and mode == "test":
        net=VNet_ssnet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    elif net_type == "vnet_mcnet" and mode == "test":
        net=MCNet3d_v2(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    elif net_type == "vnet_dtc" and mode == "test":
        net=VNet_dtc(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()  
    else:
        net = None
    return net

