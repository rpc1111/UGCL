import argparse
import logging
import os
import random
import shutil
import sys
import time

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

from dataloaders import utils
from dataloaders.brats2019 import (BraTS2019, LAHeart,CenterCrop, RandomCrop,
                                   RandomRotFlip, ToTensor,
                                   TwoStreamBatchSampler)
from networks.net_factory_3d import net_factory_3d_1
from utils import losses, metrics, ramps
from val_3D import test_all_case
import test_3d_patch

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/LA', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='LA/Mean_Teacher', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='vnet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=15000, help='maximum epoch number to train')
parser.add_argument('--max_samples', type=int,  default=80, help='maximum samples to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[112, 112, 80],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=2,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=4,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=40.0, help='consistency_rampup')

args = parser.parse_args()


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
def pixelwisecontrastiveloss(student_proj,teachar_proj,final_indices,unlabel,old_pos,epoch_num,iter_num,epoch_n_s,old,ema_output2,unlabel_ema_s):
    b_loss = 0
    label = torch.unique(unlabel)
    label = label[label != 255]
    ema_output_un=ema_output2[2:]
    inputs = student_proj.permute(1, 0, 2, 3, 4)  # c x b x h x w
    target = teachar_proj.permute(1, 0, 2, 3, 4)  # c x b x h x w
    output2 = ema_output2.permute(1, 0, 2, 3, 4)
    output2_un=ema_output_un.permute(1, 0, 2, 3,4)

    x=1.0
    n=0.01
    for idx in label:
   
        input_vec = inputs[:, ( unlabel==255)*(unlabel_ema_s==idx)*(output2_un[idx]>0.75)].T  
        pos_vec = target[:, (idx == final_indices)*(output2[idx] >= 0.9)]  
        pos_mean=pos_vec.mean(dim=1,keepdim=True)
     
        if epoch_n_s==epoch_num-1:
           old[idx]=old_pos[idx]/2.0
           old_pos[idx]=0.0
        if epoch_num>=1200:
           old_pos[idx]=old_pos[idx]+pos_mean

        if epoch_num>1200:
           pos_mean=old[idx]
           x=0.3
        if len(input_vec)<1:
            break
        
        input_vec_1=input_vec[random.sample(range(len(input_vec)),min(len(input_vec),100))]
        
        n=n+len(input_vec_1)
        input_vec = input_vec_1 / input_vec_1.norm(dim=1, keepdim=True).clamp(min=1e-8)
        pos_vec = pos_mean / pos_mean.norm(dim=0, keepdim=True).clamp(min=1e-8)
        neg_v_1 = target[:, ((idx+1)%2 == final_indices)].T  


        for i in input_vec:
            i=i.reshape(1,96)
            neg_vec_1 = neg_v_1[random.sample(range(len(neg_v_1)),min(len(neg_v_1),256))]
            if epoch_num>1205:
                neg_vec_1=neg_vec_1*x+(1-x)*(old[(idx+1)%2].reshape(1,96))
            neg_v = neg_vec_1 / neg_vec_1.norm(dim=1, keepdim=True).clamp(min=1e-8)
            pos_pair = torch.mm(i, pos_vec)
            neg_pair = torch.mm(i, neg_v.T)
            pos_pair = torch.exp(pos_pair / 0.5).sum().clamp(min=1e-8)
            neg_pair = torch.exp(neg_pair / 0.5).sum().clamp(min=1e-8)
            b_loss += -(torch.log(pos_pair / (neg_pair + pos_pair)))

    return b_loss / (n),old_pos,epoch_num,old

def entropy_Filtering3(output,unlabel,epoch_num,epoch_max):
    with torch.no_grad():
        label = torch.unique(unlabel)
        label = label[label != 255]
        k=[0.02,0.1]
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
        return unlabel
def Patch_Filtering(inputs):
    with torch.no_grad():
        inputs=inputs.reshape(2,112,112,80)
        unfold1  = torch.nn.Unfold(kernel_size=(4, 4), stride=(4,4))
        unfold2  = torch.nn.Unfold(kernel_size=(64, 1), stride=(64,1))
        patches = unfold1(inputs).reshape(2,1,1792,560)
        patches = unfold2(patches).permute(0,2,1)
        b=[]
        for i in [0,1]:
            b1=torch.sum(patches==i,dim=2)/64>=0.99
            b.append(b1)
            patches[b1]=torch.where(patches[b1]==i,patches[b1],255)
        c=b[0]+b[1]
        patches[~c]=255
        fold2 = torch.nn.Fold(output_size=(1792,560), kernel_size=(64, 1), stride=(64,1))
        fold1 = torch.nn.Fold(output_size=(112, 80), kernel_size=(4, 4), stride=(4,4))
        inputs_restore = fold2(patches.permute(0,2,1))
        inputs_restore2=fold1(inputs_restore.reshape(2,1792,560))
        return inputs_restore2

def train(args, snapshot_path):
    base_lr = args.base_lr
    train_data_path = args.root_path
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    num_classes = 2

    def create_model(ema=False):
        # Network definition
        net = net_factory_3d_1(net_type=args.model, in_chns=1, class_num=num_classes)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)

    db_train = LAHeart(base_dir=train_data_path,
                         split='train',
                         num=None,
                         transform=transforms.Compose([
                             RandomRotFlip(),
                             RandomCrop(args.patch_size),
                             ToTensor(),
                         ]))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    labeled_idxs = list(range(0, args.labeled_num))
    unlabeled_idxs = list(range(args.labeled_num, 80))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()
    ema_model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(2)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_dice = 0.0

    iterator = tqdm(range(max_epoch), ncols=70)
    old_pos=[0.0,0.0]
    epoch_n_s=1
    old=[0.0,0.0]
    for epoch_num in iterator:
        print('')
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch = volume_batch[args.labeled_bs:]
            
            noise = torch.clamp(torch.randn_like(
                unlabeled_volume_batch) * 0.05, -0.1, 0.1)
            ema_inputs = unlabeled_volume_batch + noise
            x=torch.cat([volume_batch[:args.labeled_bs], ema_inputs],dim=0)
            outputs,s_proj= model(x)
            outputs_soft = torch.softmax(outputs, dim=1)
            with torch.no_grad():
                ema_output,t_proj = ema_model(volume_batch)
                ema_output_soft = torch.softmax(ema_output, dim=1)
                unlabel_ema=ema_output_soft.argmax(1)
                ema_output_soft[:args.labeled_bs]=0.99
            loss_ce = ce_loss(outputs[:args.labeled_bs],
                              label_batch[:args.labeled_bs][:])
            loss_dice = dice_loss(
                outputs_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            supervised_loss =  (loss_dice + loss_ce)
            #consistency_weight = get_current_consistency_weight(iter_num//150)
            #consistency_loss = torch.mean(
            #    (outputs_soft[args.labeled_bs:] - ema_output_soft)**2)
            if iter_num < 2000:
                contrastiveloss=0.0
                unceloss=0.0
                consistency_loss=0.0
            else:
                unlabel=Patch_Filtering(unlabel_ema[args.labeled_bs:].float()).long()
                unlabel=entropy_Filtering3(ema_output[args.labeled_bs:],unlabel,epoch_num,max_epoch)
                un_label=torch.cat([label_batch[:][:args.labeled_bs],unlabel],dim=0)
                unceloss= nn.CrossEntropyLoss(ignore_index=255)(outputs[args.labeled_bs:],unlabel)
                contrastiveloss,old_pos,epoch_n_s,old=pixelwisecontrastiveloss(s_proj[args.labeled_bs:],t_proj,un_label,unlabel,old_pos,epoch_num,iter_num,epoch_n_s,old,ema_output_soft,unlabel_ema[args.labeled_bs:])
                with torch.no_grad():
                    unlabel_c=unlabel.clone()
                    unlabel_c=unlabel_c.reshape(2,1,112,112,80)
                    ulab=torch.cat([unlabel_c,unlabel_c],dim=1)
                    ema_output2=torch.where(ulab!=255,outputs_soft[args.labeled_bs:],ema_output_soft[args.labeled_bs:])
                consistency_loss = torch.sum(
                        (outputs_soft[args.labeled_bs:]-ema_output2)**2)/(torch.sum(unlabel_c==255)+0.01)
            
            
            loss = supervised_loss + 1.25*unceloss+0.03*contrastiveloss+0.03*consistency_loss


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
   

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f,unloss_ce:%f,contrastiveloss:%f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item(),unceloss,contrastiveloss))
            writer.add_scalar('loss/loss', loss, iter_num)

            if iter_num % 20 == 0:
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                image = outputs_soft[0, 1:2, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label',
                                 grid_image, iter_num)

                image = label_batch[0, :, :, 20:61:10].unsqueeze(
                    0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_label',
                                 grid_image, iter_num)

            if iter_num > 2000 and iter_num % 200 == 0:
                
                model.eval()
                dice_sample = test_3d_patch.var_all_case_LA(model, num_classes=num_classes, patch_size=args.patch_size, stride_xy=18, stride_z=4)
                if dice_sample > best_dice:
                    best_dice = round(dice_sample, 4)
                    save_mode_path = os.path.join(snapshot_path,  'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)
                    logging.info("save best model to {}".format(save_mode_path))
                writer.add_scalar('4_Var_dice/Dice', dice_sample, iter_num)
                writer.add_scalar('4_Var_dice/Best_dice', best_dice, iter_num)
                model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}/{}_1".format(
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
