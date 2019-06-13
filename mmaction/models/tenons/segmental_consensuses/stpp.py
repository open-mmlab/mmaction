import torch
import torch.nn as nn
import torch.nn.functional as F
from ...registry import SEGMENTAL_CONSENSUSES
import numpy as np


def parse_stage_config(stage_cfg):
    if isinstance(stage_cfg, int):
        return (stage_cfg,), stage_cfg
    elif isinstance(stage_cfg, tuple) or isinstance(stage_cfg, list):
        return stage_cfg, sum(stage_cfg)
    else:
        raise ValueError("Incorrect STPP config {}".format(stage_cfg))


@SEGMENTAL_CONSENSUSES.register_module
class StructuredTemporalPyramidPooling(nn.Module):
    def __init__(self, standalong_classifier=False, stpp_cfg=(1, (1,2), 1), num_seg=(2,5,2)):
        super(StructuredTemporalPyramidPooling, self).__init__()

        self.sc = standalong_classifier

        starting_parts, starting_mult = parse_stage_config(stpp_cfg[0])
        course_parts, course_mult = parse_stage_config(stpp_cfg[1])
        ending_parts, ending_mult = parse_stage_config(stpp_cfg[2])

        self.feat_multiplier = starting_mult + course_mult + ending_mult
        self.parts = (starting_parts, course_parts, ending_parts)
        self.norm_num = (starting_mult, course_mult, ending_mult)

        self.num_seg = num_seg

    def init_weights(self):
        pass

    def forward(self, input, scaling):
        x1 = self.num_seg[0]
        x2 = x1 + self.num_seg[1]
        n_seg = x2 + self.num_seg[2]

        feat_dim = input.size(1)
        src = input.view(-1, n_seg, feat_dim)
        num_sample = src.size(0)

        scaling = scaling.view(-1, 2)

        def get_stage_stpp(stage_feat, stage_parts, norm_num, scaling):
            stage_stpp = []
            stage_len = stage_feat.size(1)
            for n_part in stage_parts:
                ticks = torch.arange(0, stage_len + 1e-5, stage_len / n_part)
                for i in range(n_part):
                    part_feat = stage_feat[:, int(ticks[i]):int(ticks[i+1]), :].mean(dim=1) / norm_num
                    if scaling is not None:
                        part_feat = part_feat * scaling.view(num_sample, 1)
                    stage_stpp.append(part_feat)
            return stage_stpp

        feature_parts = []
        feature_parts.extend(get_stage_stpp(src[:, :x1, :], self.parts[0], self.norm_num[0], scaling[:, 0]))
        feature_parts.extend(get_stage_stpp(src[:, x1:x2, :], self.parts[1], self.norm_num[1], None))
        feature_parts.extend(get_stage_stpp(src[:, x2:, :], self.parts[2], self.norm_num[2], scaling[:, 1]))
        stpp_feat = torch.cat(feature_parts, dim=1)
        if not self.sc:
            return stpp_feat, stpp_feat
        else:
            course_feat = src[:, x1:x2, :].mean(dim=1)
            return course_feat, stpp_feat


@SEGMENTAL_CONSENSUSES.register_module
class STPPReorganized(nn.Module):
    def __init__(self, feat_dim, act_score_len,
                 comp_score_len, reg_score_len,
                 standalong_classifier=False,
                 with_regression=True,
                 stpp_cfg=(1, (1,2), 1)):
        super(STPPReorganized, self).__init__()

        self.sc = standalong_classifier
        self.feat_dim = feat_dim
        self.act_score_len = act_score_len
        self.comp_score_len = comp_score_len
        self.reg_score_len = reg_score_len
        self.with_regression = with_regression

        starting_parts, starting_mult = parse_stage_config(stpp_cfg[0])
        course_parts, course_mult = parse_stage_config(stpp_cfg[1])
        ending_parts, ending_mult = parse_stage_config(stpp_cfg[2])

        self.feat_multiplier = starting_mult + course_mult + ending_mult
        self.stpp_cfg = (starting_parts, course_parts, ending_parts)

        self.act_slice = slice(0, self.act_score_len if self.sc else (self.act_score_len * self.feat_multiplier))
        self.comp_slice = slice(self.act_slice.stop, self.act_slice.stop + self.comp_score_len * self.feat_multiplier)
        self.reg_slice = slice(self.comp_slice.stop, self.comp_slice.stop + self.reg_score_len * self.feat_multiplier)


    def init_weights(self):
        pass

    def forward(self, input, proposal_ticks, scaling):
        assert input.size(1) == self.feat_dim
        n_ticks = proposal_ticks.size(0)

        out_act_scores = torch.zeros((n_ticks, self.act_score_len)).type_as(input)
        raw_act_scores = input[:, self.act_slice]

        out_comp_scores = torch.zeros((n_ticks, self.comp_score_len)).type_as(input)
        raw_comp_scores = input[:, self.comp_slice]

        if self.with_regression:
            out_reg_scores = torch.zeros((n_ticks, self.reg_score_len)).type_as(input)
            raw_reg_scores = input[:, self.reg_slice]
        else:
            out_reg_scores = None
            raw_reg_scores = None

        def pspool(out_scores, index, raw_scores, ticks, scaling, score_len, stpp_cfg):
            offset = 0
            for stage_idx, stage_cfg in enumerate(stpp_cfg):
                if stage_idx == 0:
                    s = scaling[0]
                elif stage_idx == len(stpp_cfg) - 1:
                    s = scaling[1]
                else:
                    s = 1.0

                stage_cnt = sum(stage_cfg)
                left = ticks[stage_idx]
                right = max(ticks[stage_idx] + 1, ticks[stage_idx + 1])

                if right <= 0 or left >= raw_scores.size(0):
                    offset += stage_cnt
                    continue
                for n_part in stage_cfg:
                    part_ticks = np.arange(left, right+1e-5, (right - left) / n_part)
                    for i in range(n_part):
                        pl = int(part_ticks[i])
                        pr = int(part_ticks[i+1])
                        if pr - pl >= 1:
                            out_scores[index, :] += raw_scores[pl:pr,
                                                               offset * score_len : (offset + 1) * score_len].mean(dim=0) * s
                        offset += 1

        for i in range(n_ticks):
            ticks = proposal_ticks[i].cpu().numpy()
            if self.sc:
                out_act_scores[i, :] = raw_act_scores[ticks[1]: max(ticks[1] + 1, ticks[2]), :].mean(dim=0)
            else:
                pspool(out_act_scores, i, raw_act_scores, ticks, scaling[i], self.act_score_len, self.stpp_cfg)

            pspool(out_comp_scores, i, raw_comp_scores, ticks, scaling[i], self.comp_score_len, self.stpp_cfg)

            if self.with_regression:
                pspool(out_reg_scores, i, raw_reg_scores, ticks, scaling[i], self.reg_score_len, self.stpp_cfg)

        return out_act_scores, out_comp_scores, out_reg_scores






