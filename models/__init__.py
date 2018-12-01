from models.backbones import resnet


def build_backbone(backbone, output_stride):
    if backbone == 'resnet':
        return resnet.build_resnet101(output_stride)
    else:
        raise NotImplementedError
