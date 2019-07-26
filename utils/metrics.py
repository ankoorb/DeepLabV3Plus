import numpy as np
from tqdm import tqdm


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iou = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iou[freq > 0]).sum()
        return FWIoU

    def Mean_Dice_Coefficient(self):
        """
        IoU = Dice/(2 - Dice), REF:https://medium.com/datadriveninvestor/deep-learning-in-medical-imaging-3c1008431aaf
        Dice = (2 IoU)/(1 + IoU) 
        """
        IoU = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix))

        Dice = (2. * IoU) / (1. + IoU)
        MDice = np.nanmean(Dice)

        return MDice

    def Dice_Coefficient(self):
        """
        Use when num_class is 2
        
        Dice = 2 TP / (2 TP + FP + FN)
        """
        TP = np.diag(self.confusion_matrix)[0]
        rev_diag = np.diag(np.fliplr(self.confusion_matrix))
        Dice = (2. * TP) / (2. * TP + np.sum(rev_diag))

        return Dice

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


class DiceCoefficient(object):
    def __init__(self, thresh_step=0.1):
        self.dice_scores = []
        self.thresholds = np.arange(0.01, 1, thresh_step)

    def _compute_dice(self, preds, targs):
        n = preds.shape[0]
        preds = preds.reshape((n, -1))
        targs = targs.reshape((n, -1))
        intersect = np.sum(preds * targs, axis=1).astype('float')
        union = np.sum(preds + targs, axis=1).astype('float')
        u0 = union == 0
        intersect[u0] = 1
        union[u0] = 2
        return (2. * intersect / union)

    def compute_best_threshold(self, preds, targs):
        """
        preds: All predictions in val set (np.array)
        targs: All targets in val set (np.array)
        """
        for i in tqdm(self.thresholds):
            temp_preds = (preds > i).astype('int')
            self.dice_scores.append(np.mean(self._compute_dice(temp_preds, targs)))
        dice_scores = np.array(self.dice_scores)
        best_dice = dice_scores.max()
        best_threshold = self.thresholds[dice_scores.argmax()]
        return best_dice, best_threshold

    def _compute_dice_torch(self, preds, targs):
        n = preds.shape[0]
        preds = preds.view(n, -1)
        targs = targs.view(n, -1)
        intersect = (preds * targs).sum(-1).float()
        union = (preds + targs).sum(-1).float()
        u0 = union == 0
        intersect[u0] = 1
        union[u0] = 2
        return (2. * intersect / union)

    def compute_best_threshold_torch(self, preds, targs):
        """
        preds: All predictions in val set (Torch Tensor)
        targs: All targets in val set (Torch Tensor)
        """
        for i in tqdm(self.thresholds):
            temp_preds = (preds > i).long()
            self.dice_scores.append(self._compute_dice_torch(temp_preds, targs).mean())
        dice_scores = np.array(self.dice_scores)
        best_dice = dice_scores.max()
        best_threshold = self.thresholds[dice_scores.argmax()]
        return best_dice, best_threshold
