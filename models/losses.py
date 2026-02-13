import torch
import torch.nn.functional as F


def dice_loss_from_logits(logits, target, eps=1e-6):
    probs = torch.softmax(logits, dim=1)[:, 1]
    target = target.float()

    intersection = (probs * target).sum(dim=(1, 2))
    denom = probs.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


def segmentation_loss(logits, target, ce_weight=1.0, dice_weight=1.0):
    ce = F.cross_entropy(logits, target.long())
    dice = dice_loss_from_logits(logits, target.long())
    total = ce_weight * ce + dice_weight * dice
    return total, {"ce": ce.detach(), "dice": dice.detach()}


def masked_smooth_l1_loss(pred, target, valid_mask=None):
    if valid_mask is None:
        return F.smooth_l1_loss(pred, target)

    valid_mask = valid_mask.view(-1, 1, 1, 1).to(dtype=pred.dtype)
    per_pixel = F.smooth_l1_loss(pred, target, reduction="none")
    masked = per_pixel * valid_mask
    denom = valid_mask.sum() * pred.shape[-1] * pred.shape[-2]

    if denom.item() == 0:
        return pred.sum() * 0.0
    return masked.sum() / denom
