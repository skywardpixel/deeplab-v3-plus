from __future__ import print_function, division

import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from dataloaders import transforms as tr
from dataloaders.utils import dataset_root_dir


class VOCSegmentation(Dataset):
    """
    PascalVoc dataset
    """
    NUM_CLASSES = 21

    HUMAN_CLASS_ID = 15

    def __init__(self, args, base_dir=dataset_root_dir('pascal'), split='train'):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        """
        super(VOCSegmentation, self).__init__()
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'JPEGImages')
        self._cat_dir = os.path.join(self._base_dir, 'SegmentationClass')
        self._obj_dir = os.path.join(self._base_dir, 'SegmentationObject')
        self._human_dir = os.path.join(self._base_dir, 'SegmentationHumans')

        if not os.path.exists(self._human_dir):
            os.makedirs(self._human_dir)

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.args = args

        _splits_dir = os.path.join(self._base_dir, 'ImageSets', 'Segmentation')

        self.im_ids = []
        self.images = []
        self.categories = []

        for splt in self.split:

            if not os.path.isfile(os.path.join(_splits_dir, splt + '_multihumans.txt')):
                with open(os.path.join(_splits_dir, splt + '.txt'), "r") as f:
                    lines = f.read().splitlines()
                goodlines = []
                for ii, line in enumerate(lines):
                    _cat = Image.open(os.path.join(self._cat_dir, line + ".png")).load()
                    objimg = Image.open(os.path.join(self._obj_dir, line + ".png"))
                    _obj = objimg.load()
                    humans = {}

                    for x in range(objimg.size[0]):
                        for y in range(objimg.size[1]):
                            obj = _obj[x,y]
                            _obj[x,y] = 0 #255 if obj==255 else 0
                            if _cat[x,y] == self.HUMAN_CLASS_ID and obj!=255 and obj!=0:
                                if obj not in humans:
                                    humans[obj] = len(humans)+1
                                _obj[x,y] = humans[obj]
                    
                    if len(humans) > 1: #only requires one person in the pic now
                        if os.path.isfile(os.path.join(self._human_dir, line + ".png")):
                            os.remove(os.path.join(self._human_dir, line + ".png"))
                        objimg.save(os.path.join(self._human_dir, line + ".png"))
                        goodlines.append(line)

                with open(os.path.join(_splits_dir, splt + '_multihumans.txt'), "w") as f:
                    f.write('\n'.join(goodlines) + '\n')

            with open(os.path.join(_splits_dir, splt + '_multihumans.txt'), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                _image = os.path.join(self._image_dir, line + ".jpg")
                _cat = os.path.join(self._human_dir, line + ".png")
                assert os.path.isfile(_image)
                assert os.path.isfile(_cat)
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)

        assert (len(self.images) == len(self.categories))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}

        for split in self.split:
            if split == "train":
                return self.transform_tr(sample)
            elif split == 'val':
                return self.transform_val(sample)

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.categories[index])

        return _img, _target

    def transform_tr(self, sample):
        # composed_transforms = transforms.Compose([
        #     tr.RandomHorizontalFlip(),
        #     tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
        #     tr.RandomGaussianBlur(),
        #     tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        #     tr.ToTensor()])
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomRotate(3),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)


    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def __str__(self):
        return 'VOC2012(split=' + str(self.split) + ')'


if __name__ == '__main__':
    from dataloaders.utils import decode_segmap, dataset_root_dir, dataset_root_dir
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    voc_train = VOCSegmentation(args, split='train')

    dataloader = DataLoader(voc_train, batch_size=5, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='pascal')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)
