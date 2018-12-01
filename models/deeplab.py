import torch
import torch.nn as nn
import torch.nn.functional as F

from models import build_backbone
from models.aspp import build_aspp
from models.decoder import build_decoder


class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21, freeze_bn=False):
        super(DeepLab, self).__init__()
        self.backbone = build_backbone(backbone, output_stride)
        self.aspp = build_aspp(backbone, output_stride)
        self.decoder = build_decoder(backbone, output_stride)
        if freeze_bn:
            self.freeze_bn()

    def forward(self, x):
        input = x
        x, low_level_feat = self.backbone(x)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

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
