import torch
import torch.nn as nn

# ----- LOSS FUNCTION


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        probs = torch.softmax(predictions, dim=1)[:, 1]
        targets_f = targets.float()

        intersection = (probs * targets_f).sum(dim=(1, 2))
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum(dim=(1, 2)) + targets_f.sum(dim=(1, 2)) + self.smooth
        )

        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.5, ce_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice = DiceLoss()

        self.ce = nn.CrossEntropyLoss()

    def forward(self, predictions, targets):
        combined = self.ce_weight * self.ce(
            predictions, targets
        ) + self.dice_weight * self.dice(predictions, targets)

        return combined


def compute_metrics(predictions, targets, threshold=0.5):

    pred_labels = torch.argmax(predictions, dim=1)
    # view(-1) -> Flattens everything for into 1d vector
    pred = pred_labels.view(-1).cpu()
    true = targets.view(-1).cpu()

    tp = ((true == 1) & (pred == 1)).sum().float()
    fp = ((true == 0) & (pred == 1)).sum().float()
    fn = ((true == 1) & (pred == 0)).sum().float()

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)

    return {
        "iou": iou.item(),
        "f1": f1.item(),
        "precision": precision.item(),
        "recall": recall.item(),
    }
