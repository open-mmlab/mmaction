import torch
import torch.nn as nn
import torch.nn.functional as F
from ...registry import HEADS
from mmaction.losses import completeness_loss, classwise_regression_loss

@HEADS.register_module
class SSNHead(nn.Module):
    """SSN's classification head"""

    def __init__(self,
                 dropout_ratio=0.8,
                 in_channels_activity=3072,
                 in_channels_complete=3072,
                 num_classes=20,
                 with_bg=False,
                 with_reg=True,
		 init_std=0.001):
    
        super(SSNHead, self).__init__()

        self.dropout_ratio = dropout_ratio
        self.in_channels_activity = in_channels_activity
        self.in_channels_complete = in_channels_complete
        self.num_classes = num_classes - 1 if with_bg else num_classes
        self.with_reg = with_reg
        self.init_std = init_std

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        self.activity_fc = nn.Linear(in_channels_activity, num_classes + 1)
        self.completeness_fc = nn.Linear(in_channels_complete, num_classes)
        if self.with_reg:
            self.regressor_fc = nn.Linear(in_channels_complete, num_classes * 2)


    def init_weights(self):
        nn.init.normal_(self.activity_fc.weight, 0, self.init_std)
        nn.init.constant_(self.activity_fc.bias, 0)
        nn.init.normal_(self.completeness_fc.weight, 0, self.init_std)
        nn.init.constant_(self.completeness_fc.bias, 0)
        if self.with_reg:
            nn.init.normal_(self.regressor_fc.weight, 0, self.init_std)
            nn.init.constant_(self.regressor_fc.bias, 0)

    def prepare_test_fc(self, stpp_feat_multiplier):
        # TODO: support the case of standalone=False
        self.test_fc = nn.Linear(self.activity_fc.in_features,
                                 self.activity_fc.out_features
                                 + self.completeness_fc.out_features * stpp_feat_multiplier
                                 + (self.regressor_fc.out_features * stpp_feat_multiplier if self.with_reg else 0))
        reorg_comp_weight = self.completeness_fc.weight.data.view(
                self.completeness_fc.out_features, stpp_feat_multiplier,
                self.activity_fc.in_features).transpose(0, 1).contiguous().view(-1, self.activity_fc.in_features)
        reorg_comp_bias = self.completeness_fc.bias.data.view(1, -1).expand(
                stpp_feat_multiplier, self.completeness_fc.out_features).contiguous().view(-1) / stpp_feat_multiplier

        weight = torch.cat((self.activity_fc.weight.data, reorg_comp_weight))
        bias = torch.cat((self.activity_fc.bias.data, reorg_comp_bias))

        if self.with_reg:
            reorg_reg_weight = self.regressor_fc.weight.data.view(
                    self.regressor_fc.out_features, stpp_feat_multiplier,
                    self.activity_fc.in_features).transpose(0, 1).contiguous().view(-1, self.activity_fc.in_features)
            reorg_reg_bias = self.regressor_fc.bias.data.view(1, -1).expand(
                    stpp_feat_multiplier, self.regressor_fc.out_features).contiguous().view(-1) / stpp_feat_multiplier
            weight = torch.cat((weight, reorg_reg_weight))
            bias = torch.cat((bias, reorg_reg_bias))

        self.test_fc.weight.data = weight
        self.test_fc.bias.data = bias
        return True


    def forward(self, input, test_mode=False):
        if not test_mode:
            activity_feat, completeness_feat = input
            if self.dropout is not None:
                activity_feat = self.dropout(activity_feat)
                completeness_feat = self.dropout(completeness_feat)

            act_score = self.activity_fc(activity_feat)
            comp_score = self.completeness_fc(completeness_feat)
            bbox_pred = self.regressor_fc(completeness_feat) if self.with_reg else None

            return act_score, comp_score, bbox_pred
        else:
            test_score = self.test_fc(input)
            return test_score

    def loss(self,
             act_score,
             comp_score,
             bbox_pred,
             prop_type,
             labels,
             bbox_targets,
             train_cfg):
        losses = dict()

        prop_type = prop_type.view(-1)
        labels = labels.view(-1)
        act_indexer = ((prop_type == 0) + (prop_type == 2)).nonzero().squeeze()
        comp_indexer = ((prop_type == 0) + (prop_type == 1)).nonzero().squeeze()


        denum = train_cfg.ssn.sampler.fg_ratio + train_cfg.ssn.sampler.bg_ratio + train_cfg.ssn.sampler.incomplete_ratio
        fg_per_video = int(train_cfg.ssn.sampler.num_per_video * (train_cfg.ssn.sampler.fg_ratio / denum))
        bg_per_video = int(train_cfg.ssn.sampler.num_per_video * (train_cfg.ssn.sampler.bg_ratio / denum))
        incomplete_per_video = train_cfg.ssn.sampler.num_per_video - fg_per_video - bg_per_video

        losses['loss_act'] = F.cross_entropy(act_score[act_indexer, :],
                                             labels[act_indexer])
        losses['loss_comp'] = completeness_loss(comp_score[comp_indexer, :],
                                                labels[comp_indexer],
                                                fg_per_video,
                                                fg_per_video + incomplete_per_video,
                                                ohem_ratio=fg_per_video / incomplete_per_video)
        losses['loss_comp'] = losses['loss_comp'] * train_cfg.ssn.loss_weight.comp_loss_weight
        if bbox_pred is not None:
            reg_indexer = (prop_type == 0).nonzero().squeeze()
            bbox_targets = bbox_targets.view(-1, 2)
            bbox_pred = bbox_pred.view(-1, self.completeness_fc.out_features, 2)
            losses['loss_reg'] = classwise_regression_loss(bbox_pred[reg_indexer, :, :],
                                                           labels[reg_indexer],
                                                           bbox_targets[reg_indexer, :])
            losses['loss_reg'] = losses['loss_reg'] * train_cfg.ssn.loss_weight.reg_loss_weight
        return losses
