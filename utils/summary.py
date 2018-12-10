import os

import torch
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid

from dataloaders.utils import decode_seg_map_sequence

from models import deeplab

from PIL import Image

class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory
        self.imgz = 1

    def savethis(self,img):
        if False:
            if not os.path.exists("imgz"):
                os.makedirs("imgz")
            img=img.numpy()*255
            img=img.astype('uint8').transpose(1,2,0)
            im = Image.fromarray(img)
            fn = "imgz/"+str(self.imgz)+".png"
            if os.path.isfile(fn):
                os.remove(fn)
            im.save(fn)
            self.imgz = self.imgz +1


    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer

    def visualize_image(self, writer, dataset, image, target, output, global_step):
        AMOUNT = image.shape[0]

        grid_image = make_grid(image[:AMOUNT].clone().cpu().data, AMOUNT, normalize=True)
        writer.add_image('Image', grid_image, global_step)
        self.savethis(grid_image)

        grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(target[:AMOUNT], 1).detach().cpu().numpy(),
                                                       dataset=dataset), AMOUNT, normalize=False, range=(0, 255))
        writer.add_image('Groundtruth label', grid_image, global_step)
        self.savethis(grid_image)

        grid_image = make_grid(output[:AMOUNT,0:1].detach().cpu(), AMOUNT, normalize=False, range=(0, 255))
        writer.add_image('Is human', grid_image, global_step)
        self.savethis(grid_image)

        # chosen = deeplab.choose_objects(output, target)
        # grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(chosen[:AMOUNT], 1).detach().cpu().numpy(),
        #                                                dataset=dataset), AMOUNT, normalize=False, range=(0, 255))
        # writer.add_image('Chosen label', grid_image, global_step)
        # self.savethis(grid_image)

        for i in range(1,output.shape[1]-2,3):
            grid_image = make_grid(output[:AMOUNT,i:i+3].detach().cpu(), AMOUNT, normalize=False, range=(0, 255))
            writer.add_image('Feature Vector '+str(i)+'-'+str(i+(AMOUNT-1)), grid_image, global_step)
            self.savethis(grid_image)

        
