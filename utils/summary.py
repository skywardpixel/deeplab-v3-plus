import os

import torch
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid

from dataloaders.utils import decode_seg_map_sequence

from models import deeplab

class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer

    def visualize_image(self, writer, dataset, image, target, output, global_step):
        AMOUNT = image.shape[0]

        chosen = deeplab.choose_objects(output, target)
        grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(chosen[:AMOUNT], 1).detach().cpu().numpy(),
                                                       dataset=dataset), AMOUNT, normalize=False, range=(0, 255))
        writer.add_image('Chosen label', grid_image, global_step)

        grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(target[:AMOUNT], 1).detach().cpu().numpy(),
                                                       dataset=dataset), AMOUNT, normalize=False, range=(0, 255))
        writer.add_image('Groundtruth label', grid_image, global_step)
        

        grid_image = make_grid(image[:AMOUNT].clone().cpu().data, AMOUNT, normalize=True)
        writer.add_image('Image', grid_image, global_step)
        #grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy(),
        #                                               dataset=dataset), 3, normalize=False, range=(0, 255))


        # make sure this is clamped .clamp(0,1)
        grid_image = make_grid(output[:AMOUNT,0:1].detach().cpu(), AMOUNT, normalize=False, range=(0, 255))

        writer.add_image('Is human', grid_image, global_step)

        for i in range(1,output.shape[1]-2,3):
            grid_image = make_grid(output[:AMOUNT,i:i+3].detach().cpu(), AMOUNT, normalize=False, range=(0, 255))
            writer.add_image('Feature Vector '+str(i)+'-'+str(i+(AMOUNT-1)), grid_image, global_step)

        
