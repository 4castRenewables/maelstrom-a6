import torch


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified
    values of k.

    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def accuracy_score(y_true: int, y_pred: torch.Tensor) -> float:
    """Calculate accuracy score between true label and prediction tensor."""
    with torch.no_grad():
        return float((y_true == y_pred.argmax(1)).sum() / y_pred.shape[0])
