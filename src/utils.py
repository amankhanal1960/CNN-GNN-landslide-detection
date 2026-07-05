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

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, smooth=1e-6):
        # alpha is weight for positive class
        # gamma is focusing parameter, 2.0 is standard
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        
    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)[:,1]
        targets_f = targets.float()
        
        # p_t = probability of the true class
        pt = torch.where(targets_f == 1, probs, 1 - probs)
        
        #alpha_t = class-dependent balancing factor
        alpha_t = torch.where(targets_f == 1, self.alpha, 1 - self.alpha)
        
        # Focal Loss per pixel
        focal = -alpha_t * (1 - pt) ** self.gamma * torch.log(pt + self.smooth)
        
        return focal.mean()
        

class CombinedFocalDiceLoss(nn.Module):
    def __init__(self, focal_weight=0.5, dice_weight=0.5, alpha=0.75, gamma=2.0):
        super().__init__()
        self.focal = BinaryFocalLoss(alpha=alpha, gamma=gamma)
        self.dice = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, predictions, targets):
        combined = self.focal_weight * self.focal(
            predictions, targets
        ) + self.dice_weight * self.dice(predictions, targets)

        return combined


def compute_metrics(predictions, targets, threshold=0.5):

    probs = torch.softmax(predictions, dim=1)[:, 1]
    pred = probs > threshold

    tp = ((targets == 1) & (pred == 1)).sum().float()
    fp = ((targets == 0) & (pred == 1)).sum().float()
    fn = ((targets == 1) & (pred == 0)).sum().float()

    return tp, fp, fn
