import torch
import torch.nn.functional as F


class OHEMHingeLoss(torch.autograd.Function):
    """
    This class is the core implementation for the completeness loss in paper.
    It compute class-wise hinge loss and performs online hard negative mining
    (OHEM).
    """

    @staticmethod
    def forward(ctx, pred, labels, is_positive, ohem_ratio, group_size):
        n_sample = pred.size()[0]
        assert n_sample == len(
            labels), "mismatch between sample size and label size"
        losses = torch.zeros(n_sample)
        slopes = torch.zeros(n_sample)
        for i in range(n_sample):
            losses[i] = max(0, 1 - is_positive * pred[i, labels[i] - 1])
            slopes[i] = -is_positive if losses[i] != 0 else 0

        losses = losses.view(-1, group_size).contiguous()
        sorted_losses, indices = torch.sort(losses, dim=1, descending=True)
        keep_num = int(group_size * ohem_ratio)
        loss = torch.zeros(1).cuda()
        for i in range(losses.size(0)):
            loss += sorted_losses[i, :keep_num].sum()
        ctx.loss_ind = indices[:, :keep_num]
        ctx.labels = labels
        ctx.slopes = slopes
        ctx.shape = pred.size()
        ctx.group_size = group_size
        ctx.num_group = losses.size(0)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        labels = ctx.labels
        slopes = ctx.slopes

        grad_in = torch.zeros(ctx.shape)
        for group in range(ctx.num_group):
            for idx in ctx.loss_ind[group]:
                loc = idx + group * ctx.group_size
                grad_in[loc, labels[loc] - 1] = slopes[loc] * \
                    grad_output.data[0]
        return torch.autograd.Variable(grad_in.cuda()), None, None, None, None


def completeness_loss(pred, labels, sample_split,
                      sample_group_size, ohem_ratio=0.17):
    pred_dim = pred.size()[1]
    pred = pred.view(-1, sample_group_size, pred_dim)
    labels = labels.view(-1, sample_group_size)

    pos_group_size = sample_split
    neg_group_size = sample_group_size - sample_split
    pos_prob = pred[:, :sample_split, :].contiguous().view(-1, pred_dim)
    neg_prob = pred[:, sample_split:, :].contiguous().view(-1, pred_dim)
    pos_ls = OHEMHingeLoss.apply(pos_prob,
                                 labels[:, :sample_split].contiguous(
                                 ).view(-1), 1,
                                 1.0, pos_group_size)
    neg_ls = OHEMHingeLoss.apply(neg_prob,
                                 labels[:, sample_split:].contiguous(
                                 ).view(-1), -1,
                                 ohem_ratio, neg_group_size)
    pos_cnt = pos_prob.size(0)
    neg_cnt = int(neg_prob.size()[0] * ohem_ratio)

    return pos_ls / float(pos_cnt + neg_cnt) + \
        neg_ls / float(pos_cnt + neg_cnt)


def classwise_regression_loss(pred, labels, targets):
    indexer = labels.data - 1
    prep = pred[:, indexer, :]
    class_pred = torch.cat((torch.diag(prep[:, :, 0]).view(-1, 1),
                            torch.diag(prep[:, :, 1]).view(-1, 1)),
                           dim=1)
    loss = F.smooth_l1_loss(class_pred.view(-1), targets.view(-1)) * 2
    return loss
