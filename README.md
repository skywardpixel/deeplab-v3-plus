# deeplab-v3-plus

A PyTorch implementation of the DeepLab-v3+ model under development.
This is basically a subset of a clone of the
[pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception)
repo authored by @jfzhang95. This repo is intended for
further research on instance-level semantic segmentation.

The model was originally proposed in this paper:
[Encoder-Decoder with Atrous Separable Convolution for
Semantic Image Segmentation](https://arxiv.org/abs/1802.02611)
by Liang-Chieh Chen, et al. 

# Instructions

Download the data http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
Extract the .tar into /data so the directories are like /data/VOC2012/.....

Download the pretrained weights https://drive.google.com/open?id=1NwcwlWqA-0HqAPk3dSNNPipGMF0iS0Zu
Put the .tar inside /data

run.sh
