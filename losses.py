import torch
import torch.nn as nn


class SegmentationLosses:
    def __init__(self, weight=None, reduction='elementwise_mean', batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.reduction = reduction
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.cross_entropy
        elif mode == 'focal':
            return self.focal
        else:
            raise NotImplementedError

    def cross_entropy(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        reduction=self.reduction)
        if self.cuda:
            criterion = criterion.cuda()
        loss = criterion(logit, target.long())
        if self.batch_average:
            loss /= n
        return loss

    def focal(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        reduction=self.reduction)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt
        if self.batch_average:
            loss /= n
        return loss


if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.cross_entropy(a, b).item())
    print(loss.focal(a, b, gamma=0, alpha=None).item())
    print(loss.focal(a, b, gamma=2, alpha=0.5).item())
