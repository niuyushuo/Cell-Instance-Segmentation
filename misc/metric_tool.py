import numpy as np


class AverageMeter(object):
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def clear(self):
        self.initialized = False


class ConfuseMatrixMeter(AverageMeter):
    def __init__(self, n_class):
        super(ConfuseMatrixMeter, self).__init__()
        self.n_class = n_class

    def update_cm(self, pr, gt, weight=1):
        val = get_confuse_matrix(num_classes=self.n_class, label_gts=gt, label_preds=pr)
        self.update(val, weight)
        return cm2F1(val)

    def get_scores(self):
        return cm2score(self.sum)


def cm2F1(confusion_matrix):
    hist = confusion_matrix
    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)
    recall = tp / (sum_a1 + np.finfo(np.float32).eps)
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)
    f1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
    return np.nanmean(f1)


def cm2score(confusion_matrix):
    hist = confusion_matrix
    n_class = hist.shape[0]
    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)

    acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)
    recall = tp / (sum_a1 + np.finfo(np.float32).eps)
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)
    f1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
    mean_f1 = np.nanmean(f1)

    iou = tp / (sum_a1 + hist.sum(axis=0) - tp + np.finfo(np.float32).eps)
    mean_iou = np.nanmean(iou)

    score_dict = {"acc": acc, "miou": mean_iou, "mf1": mean_f1}
    score_dict.update(dict(zip([f"iou_{i}" for i in range(n_class)], iou)))
    score_dict.update(dict(zip([f"F1_{i}" for i in range(n_class)], f1)))
    score_dict.update(dict(zip([f"precision_{i}" for i in range(n_class)], precision)))
    score_dict.update(dict(zip([f"recall_{i}" for i in range(n_class)], recall)))
    return score_dict


def get_confuse_matrix(num_classes, label_gts, label_preds):
    def _fast_hist(label_gt, label_pred):
        mask = (label_gt >= 0) & (label_gt < num_classes)
        hist = np.bincount(
            num_classes * label_gt[mask].astype(int) + label_pred[mask],
            minlength=num_classes ** 2,
        ).reshape(num_classes, num_classes)
        return hist

    confusion_matrix = np.zeros((num_classes, num_classes))
    for lt, lp in zip(label_gts, label_preds):
        confusion_matrix += _fast_hist(lt.flatten(), lp.flatten())
    return confusion_matrix
