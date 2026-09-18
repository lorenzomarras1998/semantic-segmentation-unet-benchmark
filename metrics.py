"""
MODULO: metrics.py
DESCRIZIONE: Metriche di valutazione (Dice Score, IoU) e Loss combinata 
             (BCE + Dice Loss) per segmentazione semantica.
"""

import torch
import torch.nn as nn

def dice_score(preds, targets, threshold=0.5, smooth=1e-6):
    """
    Calcola il Dice Score per la valutazione (non differenziabile, usa soglia binaria).
    """
    probs = torch.sigmoid(preds)
    preds_bin = (probs > threshold).float()

    # Appiattisce i tensori spaziali (H, W) mantenendo batch e canali
    preds_bin = preds_bin.view(preds_bin.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds_bin * targets).sum(dim=1)
    total = preds_bin.sum(dim=1) + targets.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (total + smooth)
    return dice.mean().item()


def iou_score(preds, targets, threshold=0.5, smooth=1e-6):
    """
    Calcola l'Intersection over Union (Jaccard Index).
    """
    probs = torch.sigmoid(preds)
    preds_bin = (probs > threshold).float()

    preds_bin = preds_bin.view(preds_bin.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds_bin * targets).sum(dim=1)
    union = preds_bin.sum(dim=1) + targets.sum(dim=1) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()


class BCEDiceLoss(nn.Module):
    """
    Loss combinata ponderata: alpha * BCE + beta * SoftDice.
    Un peso maggiore su BCE aumenta la penalità per ogni singolo pixel di falso positivo (pavimenti, divani).
    """
    def __init__(self, alpha=0.6, beta=0.4, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        # 1. BCE sui logit (garantisce gradienti stabili e punisce i falsi positivi locali)
        bce_loss = self.bce(preds, targets)

        # 2. Soft Dice differenziabile
        probs = torch.sigmoid(preds)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        cardinality = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        soft_dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - soft_dice.mean()

        return (self.alpha * bce_loss) + (self.beta * dice_loss)