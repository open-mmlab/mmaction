import torch
import torch.nn.functional as F


def weighted_nll_loss(pred, label, weight, avg_factor=None):
    if avg_factor is None:
        avg_factor = max(torch.sum(weight > 0).float().item(), 1.)
    raw = F.nll_loss(pred, label, reduction='none')
    return torch.sum(raw * weight)[None] / avg_factor


def weighted_cross_entropy(pred, label, weight, avg_factor=None, reduce=True):
    if avg_factor is None:
        avg_factor = max(torch.sum(weight > 0).float().item(), 1.)
    raw = F.cross_entropy(pred, label, reduction='none')
    if reduce:
        return torch.sum(raw * weight)[None] / avg_factor
    else:
        return raw * weight / avg_factor


def weighted_binary_cross_entropy(pred, label, weight, avg_factor=None):
    if pred.dim() != label.dim():
        label, weight = _expand_binary_labels(label, weight, pred.size(-1))
    if avg_factor is None:
        avg_factor = max(torch.sum(weight > 0).float().item(), 1.)
    return F.binary_cross_entropy_with_logits(
        pred, label.float(), weight.float(),
        reduction='sum')[None] / avg_factor


def smooth_l1_loss(pred, target, beta=1.0, reduction='mean'):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, mean: 1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.sum() / pred.numel()
    elif reduction_enum == 2:
        return loss.sum()


def weighted_smoothl1(pred, target, weight, beta=1.0, avg_factor=None):
    if avg_factor is None:
        avg_factor = torch.sum(weight > 0).float().item() / 4 + 1e-6
    loss = smooth_l1_loss(pred, target, beta, reduction='none')
    return torch.sum(loss * weight)[None] / avg_factor


def accuracy(pred, target, topk=1):
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    _, pred_label = pred.topk(maxk, 1, True, True)
    pred_label = pred_label.t()
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / pred.size(0)))
    return res[0] if return_single else res


def _expand_binary_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    bin_label_weights = label_weights.view(-1, 1).expand(
        label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights


def weighted_multilabel_binary_cross_entropy(
        pred, label, weight, avg_factor=None):
    label, weight = _expand_multilabel_binary_labels(
        label, weight, pred.size(-1))
    if avg_factor is None:
        avg_factor = max(torch.sum(weight > 0).float().item(), 1.)
    return F.binary_cross_entropy_with_logits(
        pred, label.float(), weight.float(),
        reduction='sum')[None] / avg_factor


def _expand_multilabel_binary_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1)
    if inds.numel() > 0:
        for ind in inds:
            # note that labels starts from 1
            bin_labels[ind[0], labels[ind[0], ind[1]] - 1] = 1
            # bin_labels[ind[0], 0] = 1
    bin_label_weights = label_weights
    return bin_labels, bin_label_weights


def multilabel_accuracy(pred, target, topk=1, thr=0.5):
    if topk is None:
        topk = ()
    elif isinstance(topk, int):
        topk = (topk, )

    pred = pred.sigmoid()
    pred_bin_labels = pred.new_full((pred.size(0), ), 0, dtype=torch.long)
    pred_vec_labels = pred.new_full(pred.size(), 0, dtype=torch.long)
    for i in range(pred.size(0)):
        inds = torch.nonzero(pred[i, 1:] > thr).squeeze() + 1
        if inds.numel() > 0:
            pred_vec_labels[i, inds] = 1
            # pred_bin_labels[i] = 1
        if pred[i, 0] > thr:
            pred_bin_labels[i] = 1
    target_bin_labels = target.new_full(
        (target.size(0), ), 0, dtype=torch.long)
    target_vec_labels = target.new_full(target.size(), 0, dtype=torch.long)
    for i in range(target.size(0)):
        inds = torch.nonzero(target[i, :] >= 1).squeeze()
        if inds.numel() > 0:
            target_vec_labels[i, target[i, inds]] = 1
            target_bin_labels[i] = 1
    # overall accuracy
    correct = pred_bin_labels.eq(target_bin_labels)
    acc = correct.float().sum(0, keepdim=True).mul_(100.0 / correct.size(0))

    # def overlap(tensor1, tensor2):
    #     indices = tensor1.new_zeros(tensor1).astype(torch.uint8)
    #     for elem in tensor2:
    #         indices = indices | (tensor1 == elem)
    #     return tensor1[indices]

    # recall@thr
    recall_thr, prec_thr = recall_prec(pred_vec_labels, target_vec_labels)

    # recall@k
    recalls = []
    precs = []
    for k in topk:
        _, pred_label = pred.topk(k, 1, True, True)
        pred_vec_labels = pred.new_full(pred.size(), 0, dtype=torch.long)
        for i in range(pred.size(0)):
            pred_vec_labels[i, pred_label[i]] = 1
        recall_k, prec_k = recall_prec(pred_vec_labels, target_vec_labels)
        recalls.append(recall_k)
        precs.append(prec_k)

    return acc, recall_thr, prec_thr, recalls, precs


def recall_prec(pred_vec, target_vec):
    """
    Args:
        pred_vec: <torch.tensor> (n, C+1), each element is either 0 or 1
        target_vec: <torch.tensor> (n, C+1), each element is either 0 or 1

    Returns:
        recall
        prec
    """
    recall = pred_vec.new_full((pred_vec.size(0), ), 0).float()
    prec = pred_vec.new_full((pred_vec.size(0), ), 0).float()
    num_pos = 0
    for i in range(target_vec.size(0)):
        if target_vec[i, :].float().sum(0) == 0:
            continue
        correct_labels = pred_vec[i, :] & target_vec[i, :]
        recall[i] = correct_labels.float().sum(0, keepdim=True) / \
            target_vec[i, :].float().sum(0, keepdim=True)
        prec[i] = correct_labels.float().sum(0, keepdim=True) / \
            (pred_vec[i, :].float().sum(0, keepdim=True) + 1e-6)
        num_pos += 1
    recall = recall.float().sum(0, keepdim=True).mul_(100. / num_pos)
    prec = prec.float().sum(0, keepdim=True).mul_(100. / num_pos)
    return recall, prec
