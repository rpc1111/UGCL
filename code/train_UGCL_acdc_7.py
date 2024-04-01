import argparse
import logging
import os
import random
import shutil
import sys
import time
import math

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from augmentations.ctaugment import OPS
from dataloaders import utils
import augmentations
from dataloaders.dataset import (BaseDataSets, RandomGenerator,
                                 TwoStreamBatchSampler,WeakStrongAugment,CTATransform)
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_2D import test_single_volume

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/Mean_Teacher', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=12,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=7,
                    help='labeled data')

# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
args = parser.parse_args()


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]
    
def pixelwisecontrastiveloss(student_proj,teachar_proj,final_indices,unlabel,old_pos,epoch_num,epoch_n_s,old,ema_output2):
    b_loss = 0
    label = torch.unique(unlabel)
    label = label[label != 255]
    ema_output_un=ema_output2[12:]
    # mask : b x h x w
    inputs = student_proj.permute(1, 0, 2, 3)  # c x b x h x w
    target = teachar_proj.permute(1, 0, 2, 3)  # c x b x h x w
    output2 = ema_output2.permute(1, 0, 2, 3)
    output2_un=ema_output_un.permute(1, 0, 2, 3)

    x=1.0
    n=0.01
    for idx in label:
        # if idx == 0:
        #     continue
        input_vec = inputs[:, ( unlabel==255)*(output2_un[idx]>0.8)].T
 
  
        pos_vec = target[:, (idx == final_indices)*(output2[idx] >= 0.9)]  
        pos_mean=pos_vec.mean(dim=1,keepdim=True)
     
        if epoch_n_s==epoch_num-1:
           old[idx]=old_pos[idx]/11.0
           old_pos[idx]=0.0
        if epoch_num>=300:
           old_pos[idx]=old_pos[idx]+pos_mean

        if epoch_num>300:
           pos_mean=old[idx]
           x=0.1
        if len(input_vec)<5:
            break

        input_vec_1=input_vec[torch.randint(0,len(input_vec),size=(1,min(len(input_vec),100)))[0]]
        n=n+len(input_vec_1)
        input_vec = input_vec_1 / input_vec_1.norm(dim=1, keepdim=True).clamp(min=1e-8)
        pos_vec = pos_mean / pos_mean.norm(dim=0, keepdim=True).clamp(min=1e-8)
        neg_v_1 = target[:, ((idx+1)%4 == final_indices)].T  
        neg_v_2 = target[:, ((idx+2)%4 == final_indices)].T
        neg_v_3 = target[:, ((idx+3)%4 == final_indices)].T

        for i in input_vec:
            i=i.reshape(1,48)
            neg_vec_1 = neg_v_1[torch.randint(0, len(neg_v_1), size=(1, min(len(neg_v_1),80)))[0]]
            neg_vec_2 = neg_v_2[torch.randint(0, len(neg_v_2), size=(1, min(len(neg_v_2),80)))[0]]
            neg_vec_3 = neg_v_3[torch.randint(0, len(neg_v_3), size=(1, min(len(neg_v_3),80)))[0]]
            if epoch_num>305:
                neg_vec_1=neg_vec_1*x+(1-x)*(old[(idx+1)%4].reshape(1,48))
                neg_vec_2=neg_vec_2*x+(1-x)*(old[(idx+2)%4].reshape(1,48))
                neg_vec_3=neg_vec_3*x+(1-x)*(old[(idx+3)%4].reshape(1,48))
            neg_vec_1=torch.cat((neg_vec_1,neg_vec_2,neg_vec_3),dim=0)
            neg_v = neg_vec_1 / neg_vec_1.norm(dim=1, keepdim=True).clamp(min=1e-8)
            pos_pair = torch.mm(i, pos_vec)
            neg_pair = torch.mm(i, neg_v.T)
            pos_pair = torch.exp(pos_pair / 0.5).sum().clamp(min=1e-8)
            neg_pair = torch.exp(neg_pair / 0.5).sum().clamp(min=1e-8)
            b_loss += -(torch.log(pos_pair / (neg_pair + pos_pair)))


    return b_loss / (n),old_pos,epoch_num,old

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def Patch_Filtering(inputs):
    with torch.no_grad():
        inputs=inputs.reshape(12,1,256,256)
        unfold  = torch.nn.Unfold(kernel_size=(4, 4), stride=4)
        patches = unfold(inputs).permute(0,2,1)
        b=[]
        a=torch.sum(patches==0,dim=2)/16>0.99
        patches[a]=torch.where(patches[a]==0,patches[a],255)
        for i in [1,2,3]:
            b1=torch.sum(patches==i,dim=2)/16>=0.99
            b.append(b1)
            patches[b1]=torch.where(patches[b1]==i,patches[b1],255)
        b.append(a)
        c=b[0]+b[1]+b[2]+b[3]
        patches[~c]=255
        fold = torch.nn.Fold(output_size=(256, 256), kernel_size=(4, 4), stride=4)
        inputs_restore = fold(patches.permute(0,2,1))
        return inputs_restore.reshape(12,256,256)

def entropy_Filtering3(output,unlabel,epoch_num,epoch_max):
    with torch.no_grad():
        #p = F.softmax(output, dim=1)
        label = torch.unique(unlabel)
        label = label[label != 255]
        k=[0.003,0.2,0.2,0.2]
        #b=unlabel
        output_entropy = -torch.sum(F.softmax(output, dim=1) * F.log_softmax(output, dim=1), dim=1)
        for i in label:
            a=k[i]*(1-epoch_num/epoch_max)
            mask = unlabel == i
            out_en = output_entropy[mask]
            otf=out_en.flatten().cpu()
            c = np.percentile(otf, 100 * (1-a))
            c1=np.array([c])
            c2= torch.from_numpy(c1).cuda()
            mask_filter=torch.where(unlabel ==i, i, 255)
            mask_filter=torch.where(output_entropy <c2, mask_filter, 255)
            unlabel=torch.where( unlabel==i,mask_filter,unlabel)
        #print(torch.sum(unlabel==2)/torch.sum(b==2))
        return unlabel


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    #alpha = min(1 - 1 / (global_step + 1), alpha)
    alpha=0.99
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
def update_model_ema(model, ema_model, alpha):
    model_state = model.state_dict()
    model_ema_state = ema_model.state_dict()
    new_dict = {}
    for key in model_state:
        new_dict[key] = alpha * model_ema_state[key] + (1 - alpha) * model_state[key]
    ema_model.load_state_dict(new_dict)
def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    def create_model(ema=False):
        # Network definition
        model = net_factory(net_type=args.model, in_chns=1,
                            class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)
    

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None, transform=WeakStrongAugment(args.patch_size))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)


    model.train()
    #ema_model.train()

    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    old_pos=[0.0,0.0,0.0,0.0]
    epoch_n_s=1
    old=[0.0,0.0,0.0,0.0]

    for epoch_num in iterator:
        print(" ")
        for i_batch, sampled_batch in enumerate(trainloader):
            label_batch,volume_batch_w,volume_batch_s  = sampled_batch['label_aug'],sampled_batch['image_weak'],sampled_batch['image_strong']

            label_batch,volume_batch_w,volume_batch_s = label_batch.cuda(),volume_batch_w.cuda(),volume_batch_s.cuda()
            #unlabeled_volume_batch = volume_batch_w[args.labeled_bs:]

           # noise = torch.clamp(torch.randn_like(
            #unlabeled_volume_batch) * 0.1, -0.2, 0.2)
           # ema_inputs = unlabeled_volume_batch + noise

            outputs,_ = model(volume_batch_w[:args.labeled_bs])
            #outputs_noise,s_proj_noise=model(ema_inputs)
            #un_s_proj=s_proj[args.labeled_bs:]

            outputs_soft = torch.softmax(outputs, dim=1)
            with torch.no_grad():
                #ema_output,t_proj = ema_model(ema_inputs)
                ema_output1,t_proj1 = ema_model(volume_batch_w)
               # ema_output_soft= torch.softmax(ema_output, dim=1)
                ema_output2 = F.softmax(ema_output1, dim=1)
                ema_output3=ema_output2
                ema_output3[:args.labeled_bs]=0.99
                unlabel_ema=ema_output2.argmax(1)
               # ema_output_soft= torch.softmax(ema_output, dim=1)

            loss_ce = ce_loss(outputs,
                              label_batch[:][:args.labeled_bs].long())
            loss_dice = dice_loss(
                outputs_soft, label_batch[:args.labeled_bs].unsqueeze(1))
            supervised_loss =  (loss_dice + loss_ce)
            #consistency_weight = get_current_consistency_weight(iter_num//150)
            if iter_num < 3000:
                consistency_loss = 0.0
                contrastiveloss=0.0
                unceloss1=0.0
            else:
                outputs_noise,s_proj=model(volume_batch_s[args.labeled_bs:])
                outputs_soft_s = torch.softmax(outputs_noise, dim=1)
                unlabel1=Patch_Filtering(unlabel_ema[args.labeled_bs:].float()).long()
                unlabel=entropy_Filtering3(ema_output1[args.labeled_bs:],unlabel1,epoch_num,max_epoch)
                un_label=torch.cat([label_batch[:][:args.labeled_bs],unlabel],dim=0)
                contrastiveloss,old_pos,epoch_n_s,old=pixelwisecontrastiveloss(s_proj,t_proj1,un_label,unlabel,old_pos,epoch_num,epoch_n_s,old,ema_output3)
                unceloss1= nn.CrossEntropyLoss(ignore_index=255)(outputs_noise, unlabel)

                with torch.no_grad():
                    unlabel_c=unlabel.clone()
                    unlabel_c=unlabel_c.reshape(12,1,256,256)
                    ulab=torch.cat([unlabel_c,unlabel_c,unlabel_c,unlabel_c],dim=1)
                    ema_output2=torch.where(ulab!=255,outputs_soft_s,ema_output2[args.labeled_bs:])
                    
                consistency_loss = torch.sum(
                        (outputs_soft_s-ema_output2)**2)/(torch.sum(unlabel_c==255)+0.01)
                 

           
            loss = supervised_loss +1.25*unceloss1+0.15*contrastiveloss+0.1*consistency_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_model_ema(model, ema_model, 0.99)

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/consistency_loss',
                              consistency_loss, iter_num)

            writer.add_scalar('info/contrastive_loss',
                              contrastiveloss, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f ,con_loss: %f,unceloss:%f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item(),contrastiveloss,unceloss1))

           # if iter_num % 20 == 0:
               # image = volume_batch[1, 0:1, :, :]
               # writer.add_image('train/Image', image, iter_num)
               # outputs = torch.argmax(torch.softmax(
                #    outputs, dim=1), dim=1, keepdim=True)
               # writer.add_image('train/Prediction',
                 #                outputs[1, ...] * 50, iter_num)
                #labs = label_batch[1, ...].unsqueeze(0) * 50
               # writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 10000 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)
                if performance>0.896:
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}_s.pth'.format(
                                                      iter_num, round(performance, 4)))

                    torch.save(model.state_dict(), save_mode_path)
                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()

            # if iter_num % 3000 == 0:
            #     save_mode_path = os.path.join(
            #         snapshot_path, 'iter_' + str(iter_num) + '.pth')
            #     torch.save(model.state_dict(), save_mode_path)
            #     logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    #torch.use_deterministic_algorithms(True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}_labeled/{}_1".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
