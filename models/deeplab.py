import torch
import torch.nn as nn
import torch.nn.functional as F

from models import build_backbone
from models.aspp import build_aspp
from models.decoder import build_decoder

import math

import scipy.cluster.vq as vq
import numpy as np


#DO NOT SET TO EXACTLY 20
FEATUREVECTORSIZE = 30

kernel_stick_out = 6
def make_gaussian():
    kernel_size = 2*kernel_stick_out + 1
    sigma = kernel_stick_out/3.0

    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel.view(1, 1, 1, kernel_size, kernel_size)
    # gaussian_kernel = gaussian_kernel.repeat(1, 1, 1, 1)
    gaussian_filter = nn.Conv3d(in_channels=1, out_channels=1,
                                kernel_size=(1,kernel_size,kernel_size))
    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    return gaussian_filter

#target is only for KNN K
def choose_objects(output, target):
    output = output.data
    
    #is_human to a power
    importance = torch.pow(output[:,0:1],5.0)
    wfvs = importance*output[:,1:]

    gaussian = make_gaussian()
    gaussian = gaussian.cuda()
    wfvsg = gaussian(F.pad(wfvs,(kernel_stick_out,kernel_stick_out,kernel_stick_out,kernel_stick_out),mode='reflect').unsqueeze(1))
    importanceg = gaussian(F.pad(importance,(kernel_stick_out,kernel_stick_out,kernel_stick_out,kernel_stick_out),mode='reflect').unsqueeze(1))
    nfvsg = wfvsg/importanceg

    nfvsg = nfvsg.squeeze(1).data

    chosen = torch.zeros(output.shape[0], output.shape[2], output.shape[3], dtype=output.dtype)
    for img in range(output.shape[0]):
        true_k = 0
        highest = torch.max(target[img]).item()
        for person in range(1,int(highest+1)):
            if torch.sum(torch.eq(target[img], person)) > 500: #need at least X pixels
                true_k = true_k + 1
        if true_k == 0:
            continue
        fvs = []
        for x in range(output.shape[2]):
            for y in range(output.shape[3]):
                if output[img,0,x,y]>0.5:
                    fvs.append(nfvsg[img,:,x,y])
        if len(fvs) < true_k:
            continue

        fvs = torch.stack(fvs).cpu().numpy()
        centroids, labels = vq.kmeans2(fvs, fvs[np.random.choice(fvs.shape[0],true_k, False)])
        labelidx = 0
        for x in range(output.shape[2]):
            for y in range(output.shape[3]):
                if output[img,0,x,y]>0.5:
                    chosen[img,x,y] = labels[labelidx].item()+1
                    labelidx = labelidx+1
    return chosen
    

def compute_loss(output, target):
    is_human = target.clamp(0,1)
    criterion = nn.MSELoss()
    loss = criterion(output[:,0], is_human)

    #add fv loss
    for img in range(output.shape[0]):
        fvs = output[img][1:]
        these_spots = []
        these_fvs = []
        centroids = []
        total_centroid = None
        summ = 1

        highest = torch.max(target[img]).item()
        
        for person in range(1,int(highest+1)):
            spots = torch.eq(target[img], person).unsqueeze(0).float().data
            summ = spots.sum()

            if summ > 0:
                this_fvs = spots * fvs
                centroid = torch.sum(torch.sum(this_fvs.data, dim=1), dim=1)/summ
                these_spots.append(spots)
                these_fvs.append(this_fvs)
                centroids.append(centroid)
                if total_centroid is None:
                    total_centroid = centroid
                else:
                    total_centroid = total_centroid + centroid

        for c in range(len(centroids)):
            centroid = centroids[c]
            spots = these_spots[c]
            if len(centroids) > 1:
                avg_other_centroids = (total_centroid-centroid)/(len(centroids)-1)
                offset = (centroid - avg_other_centroids)
                offset = offset / torch.sum(offset*offset)
                centroid = centroid + offset #could multiply offset to balance between pulling together within groups and pushing apart different groups

                #this doesnt work as well
                #this_fvs_offset = these_fvs[c] - avg_other_centroids.view(-1,1,1).repeat(1,spots.shape[1], spots.shape[2]) * spots
                #loss = loss + 1.0/torch.sum(this_fvs_offset*this_fvs_offset)
            this_centroids = centroid.view(-1,1,1).repeat(1,spots.shape[1], spots.shape[2]) * spots
            loss = loss + criterion(these_fvs[c], this_centroids)

    return loss

class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21, freeze_bn=False):
        super(DeepLab, self).__init__()
        self.backbone = build_backbone(backbone, output_stride)
        self.aspp = build_aspp(backbone, output_stride)
        self.decoder = build_decoder(num_classes, backbone)
        if freeze_bn:
            self.freeze_bn()

    def is_project(self):
        return self.decoder.last_conv[8].out_channels!=21

    def convert_to_project(self):
        #DO NOT USE 21 CHANNEL OUTPUT FOR PROJECT
        if self.is_project():
            return

        # p(human)
        new_final_conv = nn.Conv2d(256, 1 + FEATUREVECTORSIZE, kernel_size=1, stride=1)
        
        self.decoder.last_conv[8] = new_final_conv

    def forward(self, x):
        input = x
        x, low_level_feat = self.backbone(x)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        if self.is_project():
            #x[:,0] = F.sigmoid(x[:,0])
            x = F.sigmoid(x)
        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


if __name__ == "__main__":
    model = DeepLab(backbone='resnet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())
