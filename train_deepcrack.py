import os
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms
from loss import DiceLoss

import numpy as np
import glob
import os
import argparse

from dataloader_uu3net import Rescale
from dataloader_uu3net import ToTensor
from dataloader_uu3net import CrackDataset
from dataloader_uu3net import RandomFlip

from u2net_self import U2NET

bce_loss = nn.BCELoss(size_average=True)
dice_loss = DiceLoss()

def multi_mixed_loss_fusion(d0, d9, d8, d7, d6, d5, labels):

    loss0 = bce_loss(d0, labels)
    loss9 = bce_loss(d9, labels)
    loss8 = bce_loss(d8, labels)
    loss7 = bce_loss(d7, labels)
    loss6 = bce_loss(d6, labels)
    loss5 = bce_loss(d5, labels)
    dice0 = dice_loss(d0, labels)

    loss = loss0+ loss8 +loss7 +loss6 + loss5
    loss = 0.4* dice0 + loss
    print("l0: %3f, l9: %3f, l8: %3f, l7: %3f, l6: %3f, l5: %3f\n"%(loss0.data.item(),loss9.data.item(),loss8.data.item(),loss7.data.item(),loss6.data.item(),loss5.data.item()))

    return loss0, loss


def multi_bce_loss_fusion(d0, d9, d8, d7, d6, d5, labels):

    loss0 = bce_loss(d0, labels)
    loss9 = bce_loss(d9, labels)
    loss8 = bce_loss(d8, labels)
    loss7 = bce_loss(d7, labels)
    loss6 = bce_loss(d6, labels)
    loss5 = bce_loss(d5, labels)

    loss = loss0 + loss9 + loss8 +loss7 +loss6 + loss5
    print("l0: %3f, l9: %3f, l8: %3f, l7: %3f, l6: %3f, l5: %3f\n"%(loss0.data.item(),loss9.data.item(),loss8.data.item(),loss7.data.item(),loss6.data.item(),loss5.data.item()))
    return loss0, loss

def train(args):
    epoch_num = 200
    batch_size_train = 12
    batch_size_val = 1
    train_num = 0
    val_num = 0
    model_name = 'u2net_deepcrack_fromscratch'
    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + os.sep)
    dataset_dir = "/home/uas-dtu/Documents/Chirag/CrackDetection/Datasets/deepcrack/train"
    paths = {"img_path":os.path.join(dataset_dir, "images"), "mask_path": os.path.join(dataset_dir, "masks")}
    crack_dataset = CrackDataset(
        img_path = paths['img_path'],
        mask_path = paths['mask_path'],
        transforms = transforms.Compose([
            Rescale(288),
            RandomFlip(),
            ToTensor(flag = 0)
        ]))
    crack_dataloader = DataLoader(crack_dataset, batch_size=batch_size_train, shuffle=True, num_workers=1)

    net = U2NET()
    if torch.cuda.is_available():
        net.cuda()

    print("---define optimizer...")
    net, optimizer = get_model(args)
    # ------- 5. training process --------
    print("---start training...")
    ite_num = 20000#starting iteration
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0  #starting iteration
    save_frq = 1000 # save the model every 2000 iterations

    for epoch in range(0, epoch_num):
        net.train()
        if ite_num >= 208000:
            print("finished_training")
            break

        for i, data in enumerate(crack_dataloader):
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1

            inputs, labels = data['image'], data['mask']

            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                            requires_grad=False)
            else:
                inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

            # y zero the parameter gradients
            optimizer.zero_grad()
            assert labels_v is not None
            # forward + backward + optimize
            d0, d9, d8, d7, d6, d5 = net(inputs_v)

            loss2, loss = multi_bce_loss_fusion(d0, d9, d8, d7, d6, d5, labels_v)

            loss.backward()
            optimizer.step()

            # # print statistics
            running_loss += loss.data.item()
            running_tar_loss += loss2.data.item()

            # del temporary outputs and loss
            del d0, d9, d8, d7, d6, d5, loss, loss2

            print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
            epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))

            if ite_num % save_frq == 0:

                
                torch.save(net.state_dict(), model_dir + model_name+"_bce_itr_%d_train_%3f_tar_%3f.pth" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
                print("hogya sve")
                running_loss = 0.0
                running_tar_loss = 0.0
                net.train()  # resume train
                ite_num4val = 0

def parse_args():
    parser = argparse.ArgumentParser(description = 'U2NET')
    parser.add_argument(
        '--model', type = str, default= "saved_models/u2net_deepcrack_fromscratch/u2net_deepcrack_fromscratch_bce_itr_20000_train_0.742561_tar_0.003616.pth", #model name
        help = 'model to be retrained'
    )
    args = parser.parse_args()
    return args

def get_model(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.model:
        model_path = f"{args.model}"
        model_dict = torch.load(model_path)
    net = U2NET().to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    if args.model:
        net.load_state_dict(model_dict)
        optimizer.load_state_dict(optimizer.state_dict())
        print(f"{args.model} successfully loaded")
    return net, optimizer

if __name__ == "__main__":
    args = parse_args()
    train(args)