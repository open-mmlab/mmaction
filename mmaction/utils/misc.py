import functools
import numpy as np
import mmcv


def rsetattr(obj, attr, val):
    '''
        See:
        https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects
    '''
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def rhasattr(obj, attr, *args):
    def _hasattr(obj, attr):
        if hasattr(obj, attr):
            return getattr(obj, attr)
        else:
            return None
    return functools.reduce(_hasattr, [obj] + attr.split('.')) is not None


def tensor2video_snaps(tensor, mean=(0, 0, 0), std=(1, 1, 1), to_rgb=True):
    num_videos = tensor.size(0)
    num_frames = tensor.size(2)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    video_snaps = []
    for vid_id in range(num_videos):
        img = tensor[vid_id, :, num_frames //
                     2, ...].cpu().numpy().transpose(1, 2, 0)
        img = mmcv.imdenormalize(
            img, mean, std, to_bgr=to_rgb).astype(np.uint8)
        video_snaps.append(np.ascontiguousarray(img))
    return video_snaps


def multi_apply(func, *args, **kwargs):
    pfunc = functools.partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))
