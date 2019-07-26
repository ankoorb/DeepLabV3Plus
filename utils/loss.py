import torch
import torch.nn as nn
import torch.nn.functional as F


class _DiceLoss(nn.Module):
    def __init__(self, smooth=0, eps=1e-7):
        super().__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, logit, target):
        return 1 - (2 * torch.sum(logit * target) + self.smooth) / \
                        (torch.sum(logit) + torch.sum(target) + self.smooth + self.eps)


class _MixedDiceCrossEntropyLoss(nn.Module):
    def __init__(self, dice_weight=0.2, bce_weight=0.9):
        super().__init__()
        self.dice_loss = _DiceLoss()
        self.bce_loss = nn.BCELoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, logit, target):
        pred = torch.sigmoid(logit)
        loss = self.dice_weight * self.dice_loss(pred, target) + self.bce_weight * self.bce_loss(pred, target)
        return loss


def LovaszGrad(gt_sorted):
    """
    Ref: https://github.com/nyoki-mtl/pytorch-segmentation/blob/master/src/losses/binary/lovasz_loss.py
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1 - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def Hinge(logit, target):
    signs = 2 * target - 1
    errors = 1 - logit * signs
    return errors


def LovaszHingeFlat(logit, target, ignore_index):
    """
    Binary Lovasz hinge loss
      logit: [P] Variable, logits at each prediction (between -\infty and +\infty)
      target: [P] Tensor, binary ground truth labels (0 or 1)
      ignore_index: label to ignore
    """
    logit = logit.contiguous().view(-1)
    target = target.contiguous().view(-1)
    if ignore_index is not None:
        mask = target != ignore_index
        logit = logit[mask]
        target = target[mask]
    errors = Hinge(logit, target)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = target[perm]
    grad = LovaszGrad((gt_sorted))
    loss = torch.dot(F.elu(errors_sorted) + 1, grad)
    return loss


class _LovaszLoss(nn.Module):
    """
    Binary Lovasz hinge loss
      logit: [P] Variable, logits at each prediction (between -\infty and +\infty)
      target: [P] Tensor, binary ground truth labels (0 or 1)
      ignore_index: label to ignore
    """
    def __init__(self, ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logit, target):
        return LovaszHingeFlat(logit, target, self.ignore_index)


class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == 'dice':
            return self.DiceLoss
        elif mode == 'sdice':
            return self.SmoothDiceLoss
        elif mode == 'mdice':
            return self.MixedDiceCrossEntropyLoss
        elif mode == 'lovasz':
            return self.LovaszLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

    def DiceLoss(self, logit, target):
        n, c, h, w = logit.size()
        logit = logit[:, 1, :, :]
        criterion = _DiceLoss()
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target)
        if self.batch_average:
            loss /= n

        return loss

    def SmoothDiceLoss(self, logit, target):
        n, c, h, w = logit.size()
        logit = logit[:, 1, :, :]
        criterion = _DiceLoss(smooth=1)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target)
        if self.batch_average:
            loss /= n

        return loss

    def MixedDiceCrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        logit = logit[:, 1, :, :]
        criterion = _MixedDiceCrossEntropyLoss()
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target)
        if self.batch_average:
            loss /= n

        return loss

    def LovaszLoss(self, logit, target):
        n, c, h, w = logit.size()
        logit = logit[:, 1, :, :]
        criterion = _LovaszLoss()
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target)
        if self.batch_average:
            loss /= n

        return loss


if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())

