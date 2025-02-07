def dice_score(preds, targets):
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    dice = (2.0 * intersection) / (preds.sum() + targets.sum() + 1e-7)
    return dice.item()

def iou_score(preds, targets):
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    iou = intersection / (union + 1e-7)
    return iou.item()