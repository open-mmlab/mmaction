import numpy as np
from .accuracy import softmax
import pandas as pd
from multiprocessing import Pool
import mmcv

try:
    import sys
    import os.path as osp
    sys.path.append(
        osp.abspath(osp.join(__file__, '../../../',
                             'third_party/ActivityNet/Evaluation')))
    from mmaction.third_party.ActivityNet.Evaluation.eval_detection import (
        compute_average_precision_detection)
except ImportError:
    print('Failed to import ActivityNet evaluation toolbox. Did you clone with'
          '"--recursive"?')


def results2det(dataset, outputs,
                top_k=2000, nms=0.2,
                softmax_before_filter=True,
                cls_score_dict=None,
                cls_top_k=2):
    num_class = outputs[0][1].shape[1] - 1
    detections = [dict() for i in range(num_class)]

    for idx in range(len(dataset)):
        video_id = dataset.video_infos[idx].video_id
        rel_prop = outputs[idx][0]
        if len(rel_prop[0].shape) == 3:
            rel_prop = np.squeeze(rel_prop, 0)

        act_scores = outputs[idx][1]
        comp_scores = outputs[idx][2]
        reg_scores = outputs[idx][3]
        if reg_scores is None:
            reg_scores = np.zeros(
                len(rel_prop), num_class, 2, dtype=np.float32)
        reg_scores = reg_scores.reshape((-1, num_class, 2))

        if top_k <= 0 and cls_score_dict is None:
            combined_scores = softmax(act_scores[:, 1:]) * np.exp(comp_scores)
            for i in range(num_class):
                loc_scores = reg_scores[:, i, 0][:, None]
                dur_scores = reg_scores[:, i, 1][:, None]
                detections[i][video_id] = np.concatenate((
                    rel_prop, combined_scores[:, i][:, None],
                    loc_scores, dur_scores), axis=1)
        elif cls_score_dict is None:
            combined_scores = softmax(act_scores[:, 1:]) * np.exp(comp_scores)
            keep_idx = np.argsort(combined_scores.ravel())[-top_k:]
            for k in keep_idx:
                cls = k % num_class
                prop_idx = k // num_class
                new_item = [rel_prop[prop_idx, 0], rel_prop[prop_idx, 1],
                            combined_scores[prop_idx, cls],
                            reg_scores[prop_idx, cls, 0],
                            reg_scores[prop_idx, cls, 1]]
                if video_id not in detections[cls]:
                    detections[cls][video_id] = np.array([new_item])
                else:
                    detections[cls][video_id] = np.vstack(
                        [detections[cls][video_id], new_item])
        else:
            cls_score_dict = mmcv.load(cls_score_dict)
            if softmax_before_filter:
                combined_scores = softmax(
                    act_scores[:, 1:]) * np.exp(comp_scores)
            else:
                combined_scores = act_scores[:, 1:] * np.exp(comp_scores)
            video_cls_score = cls_score_dict[video_id]

            for video_cls in np.argsort(video_cls_score,)[-cls_top_k:]:
                loc_scores = reg_scores[:, video_cls, 0][:, None]
                dur_scores = reg_scores[:, video_cls, 1][:, None]
                detections[video_cls][video_id] = np.concatenate((
                    rel_prop, combined_scores[:, video_cls][:, None],
                    loc_scores, dur_scores), axis=1)

    return detections


def perform_regression(detections):
    t0 = detections[:, 0]
    t1 = detections[:, 1]
    center = (t0 + t1) / 2
    duration = t1 - t0

    new_center = center + duration * detections[:, 3]
    new_duration = duration * np.exp(detections[:, 4])

    new_detections = np.concatenate((
        np.clip(new_center - new_duration / 2, 0, 1)[:, None],
        np.clip(new_center + new_duration / 2, 0, 1)[:, None],
        detections[:, 2:]), axis=1)
    return new_detections


def temporal_nms(detections, thresh):
    t0 = detections[:, 0]
    t1 = detections[:, 1]
    scores = detections[:, 2]

    durations = t1 - t0
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        tt0 = np.maximum(t0[i], t0[order[1:]])
        tt1 = np.minimum(t1[i], t1[order[1:]])
        intersection = tt1 - tt0
        iou = intersection / \
            (durations[i] + durations[order[1:]] - intersection).astype(float)

        inds = np.where(iou <= thresh)[0]
        order = order[inds + 1]

    return detections[keep, :]


def det2df(detections, cls):
    detection_list = []
    for vid, dets in detections[cls].items():
        detection_list.extend([[vid, cls] + x[:3] for x in dets.tolist()])
    df = pd.DataFrame(detection_list, columns=[
                      'video-id', 'cls', 't-start', 't-end', 'score'])
    return df


def eval_ap(iou, iou_idx, cls, gt, prediction):
    ap = compute_average_precision_detection(gt, prediction, iou)
    sys.stdout.flush()
    return cls, iou_idx, ap


def eval_ap_parallel(detections, gt_by_cls, iou_range, worker=32):
    ap_values = np.zeros((len(detections), len(iou_range)))

    def callback(rst):
        sys.stdout.flush()
        ap_values[rst[0], rst[1]] = rst[2][0]

    pool = Pool(worker)
    jobs = []
    for iou_idx, min_overlap in enumerate(iou_range):
        for cls in range(len(detections)):
            jobs.append(pool.apply_async(eval_ap, args=([min_overlap], iou_idx,
                                                        cls, gt_by_cls[cls],
                                                        detections[cls],),
                                         callback=callback))
    pool.close()
    pool.join()
    return ap_values
