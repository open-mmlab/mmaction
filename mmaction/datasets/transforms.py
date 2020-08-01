import mmcv
import numpy as np
import random
import math
import cv2
import random as rd

__all__ = ['GroupImageTransform', 'ImageTransform', 'BboxTransform']

class GroupColorJitter(object):
    def __init__(self, color_space_aug=False, alphastd=0.1, eigval=None, eigvec=None):
        if eigval is None:
            # note that the data range should be [0, 255]
            self.eigval = np.array([55.46, 4.794, 1.148])
        if eigvec is None:
            self.eigvec = np.array([[-0.5675, 0.7192, 0.4009],
                           [-0.5808, -0.0045, -0.8140],
                           [-0.5836, -0.6948, 0.4203]])
        self.alphastd = alphastd
        self.color_space_aug = color_space_aug

    @staticmethod
    def brightnetss(img, delta):
        if random.uniform(0, 1) > 0.5:
           #delta = np.random.uniform(-32, 32)
           delta = np.array(delta).astype(np.float32)
           img = img + delta
           #img_group = [img + delta for img in img_group]
        return img

    @staticmethod
    def contrast(img, alpha):
        if random.uniform(0, 1) > 0.5:
           #alpha = np.random.uniform(0.6,1.4)
           alpha = np.array(alpha).astype(np.float32)
           img = img * alpha
           #img_group = [img * alpha for img in img_group]
        return img

    @staticmethod
    def saturation(img, alpha):
        if random.uniform(0, 1) > 0.5:
           #alpha = np.random.uniform(0.6,1.4)
           gray = img * np.array([0.299, 0.587, 0.114]).astype(np.float32)
           gray = np.sum(gray, 2, keepdims=True)
           gray *= (1.0 - alpha)
           img = img * alpha
           img = img + gray
        return img

    @staticmethod
    def hue(img, alpha):
        if random.uniform(0, 1) > 0.5:
           #alpha = random.uniform(-18, 18)
           u = np.cos(alpha * np.pi)
           w = np.sin(alpha * np.pi)
           bt = np.array([[1.0, 0.0, 0.0],
                           [0.0, u, -w],
                           [0.0, w, u]])
           tyiq = np.array([[0.299, 0.587, 0.114],
                             [0.596, -0.274, -0.321],
                             [0.211, -0.523, 0.311]])
           ityiq = np.array([[1.0, 0.956, 0.621],
                              [1.0, -0.272, -0.647],
                              [1.0, -1.107, 1.705]])
           t = np.dot(np.dot(ityiq, bt), tyiq).T
           t = np.array(t).astype(np.float32)
           img = np.dot(img, t)
           #img_group = [np.dot(img, t) for img in img_group]
        return img

    def __call__(self, img_group):
        if self.color_space_aug:
            bright_delta = np.random.uniform(-32, 32)
            contrast_alpha = np.random.uniform(0.6,1.4)
            saturation_alpha = np.random.uniform(0.6,1.4)
            hue_alpha = random.uniform(-18, 18)
            out = []
            for img in img_group:
                img = self.brightnetss(img, delta=bright_delta)
                if random.uniform(0, 1) > 0.5:
                    img = self.contrast(img, alpha=contrast_alpha)
                    img = self.saturation(img, alpha=saturation_alpha)
                    img = self.hue(img, alpha=hue_alpha)
                else:
                    img = self.saturation(img, alpha=saturation_alpha)
                    img = self.hue(img, alpha=hue_alpha)
                    img = self.contrast(img, alpha=contrast_alpha)
                out.append(img)
            img_group = out

        alpha = np.random.normal(0, self.alphastd, size=(3,))
        rgb = np.array(np.dot(self.eigvec * alpha, self.eigval)).astype(np.float32)
        bgr = np.expand_dims(np.expand_dims(rgb[::-1], 0),0)
        return [img + rgb for img in img_group]

class RandomResizedCrop(object):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)):
        self.size = size
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        for attempt in range(10):
            area = img.shape[0] * img.shape[1]
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.shape[0] and h <= img.shape[1]:
                i = random.randint(0, img.shape[1] - h)
                j = random.randint(0, img.shape[0] - w)
                return i, j, h, w

        # Fallback
        w = min(img.shape[0], img.shape[1])
        i = (img.shape[1] - w) // 2
        j = (img.shape[0] - w) // 2
        return i, j, w, w

    def __call__(self, img_group):
        """
        Args:
            clip (list of PIL Image): list of Image to be cropped and resized.
        Returns:
            list of PIL Image: Randomly cropped and resized image.
        """
        x1, y1, th, tw = self.get_params(img_group[0], self.scale, self.ratio)
        box = np.array([x1, y1, x1+tw-1, y1+th-1], dtype=np.float32)
        return ([mmcv.imresize(mmcv.imcrop(img, box), self.size) for img in img_group], box)



class RandomRescaledCrop(object):
    def __init__(self, size, scale=(256, 320)):
        self.size = size
        self.scale = scale

    def __call__(self, img_group):
        shortedge = float(random.randint(*self.scale))

        w, h, _ = img_group[0].shape
        scale = max(shortedge / w, shortedge / h)
        img_group = [mmcv.imrescale(img, scale) for img in img_group]
        w, h, _ = img_group[0].shape
        w_offset = random.randint(0, w - self.size[0])
        h_offset = random.randint(0, h - self.size[1])

        box = np.array([w_offset, h_offset,
                        w_offset + self.size[0] - 1, h_offset + self.size[1] - 1],
                        dtype=np.float32)

        return ([img[w_offset: w_offset + self.size[0], h_offset: h_offset + self.size[1]] for img in img_group], box)


class GroupCrop(object):
    def __init__(self, crop_quadruple):
        self.crop_quadruple = crop_quadruple

    def __call__(self, img_group, is_flow=False):
        return [mmcv.imcrop(img, self.crop_quadruple)
                for img in img_group], self.crop_quadruple


class GroupCenterCrop(object):
    def __init__(self, size):
        self.size = size if not isinstance(size, int) else (size, size)

    def __call__(self, img_group, is_flow=False):
        h = img_group[0].shape[0]
        w = img_group[0].shape[1]
        tw, th = self.size
        x1 = (w - tw) // 2
        y1 = (h - th) // 2
        box = np.array([x1, y1, x1+tw-1, y1+th-1])
        return ([mmcv.imcrop(img, box) for img in img_group],
                np.array([x1, y1, tw, th], dtype=np.float32))


class Group3CropSample(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size if not isinstance(
            crop_size, int) else (crop_size, crop_size)

    def __call__(self, img_group, is_flow=False):

        image_h = img_group[0].shape[0]
        image_w = img_group[0].shape[1]
        crop_w, crop_h = self.crop_size
        assert crop_h == image_h or crop_w == image_w

        if crop_h == image_h:
            w_step = (image_w - crop_w) // 2
            offsets = list()
            offsets.append((0, 0))  # left
            offsets.append((2 * w_step, 0))  # right
            offsets.append((w_step, 0))  # middle
        elif crop_w == image_w:
            h_step = (image_h - crop_h) // 2
            offsets = list()
            offsets.append((0, 0))  # top
            offsets.append((0, 2 * h_step))  # down
            offsets.append((0, h_step))  # middle

        oversample_group = list()
        for o_w, o_h in offsets:
            normal_group = list()
            flip_group = list()
            for i, img in enumerate(img_group):
                crop = mmcv.imcrop(img, np.array(
                    [o_w, o_h, o_w + crop_w-1, o_h + crop_h-1]))
                normal_group.append(crop)
                flip_crop = mmcv.imflip(crop)

                if is_flow and i % 2 == 0:
                    flip_group.append(mmcv.iminvert(flip_crop))
                else:
                    flip_group.append(flip_crop)

            oversample_group.extend(normal_group)
            # oversample_group.extend(flip_group)
        return oversample_group, None


class GroupOverSample(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size if not isinstance(
            crop_size, int) else (crop_size, crop_size)

    def __call__(self, img_group, is_flow=False):

        image_h = img_group[0].shape[0]
        image_w = img_group[0].shape[1]
        crop_w, crop_h = self.crop_size

        offsets = GroupMultiScaleCrop.fill_fix_offset(
            False, image_w, image_h, crop_w, crop_h)
        oversample_group = list()
        for o_w, o_h in offsets:
            normal_group = list()
            flip_group = list()
            for i, img in enumerate(img_group):
                crop = mmcv.imcrop(img, np.array(
                    [o_w, o_h, o_w + crop_w-1, o_h + crop_h-1]))
                normal_group.append(crop)
                flip_crop = mmcv.imflip(crop)

                if is_flow and i % 2 == 0:
                    flip_group.append(mmcv.iminvert(flip_crop))
                else:
                    flip_group.append(flip_crop)

            oversample_group.extend(normal_group)
            oversample_group.extend(flip_group)
        return oversample_group, None


class GroupMultiScaleCrop(object):

    def __init__(self, input_size,
                 scales=None, max_distort=1,
                 fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, .875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [
            input_size, input_size]
        self.interpolation = 'bilinear'

    def __call__(self, img_group, is_flow=False):

        im_h = img_group[0].shape[0]
        im_w = img_group[0].shape[1]

        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(
            (im_w, im_h))
        box = np.array([offset_w, offset_h, offset_w +
                        crop_w - 1, offset_h + crop_h - 1])
        crop_img_group = [mmcv.imcrop(img, box) for img in img_group]
        ret_img_group = [mmcv.imresize(
            img, (self.input_size[0], self.input_size[1]),
            interpolation=self.interpolation)
            for img in crop_img_group]
        return (ret_img_group, np.array([offset_w, offset_h, crop_w, crop_h],
                                        dtype=np.float32))

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.input_size[1] if abs(
            x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(
            x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(
                image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(
            self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret


class GroupImageTransform(object):
    """Preprocess a group of images.
    1. rescale the images to expected size
    2. (for classification networks) crop the images with a given size
    3. flip the images (if needed)
    4(a) divided by 255 (0-255 => 0-1, if needed)
    4. normalize the images
    5. pad the images (if needed)
    6. transpose to (c, h, w)
    7. stack to (N, c, h, w)
    where, N = 1 * N_oversample * N_seg * L
    """

    def __init__(self,
                 mean=(0, 0, 0),
                 std=(1, 1, 1),
                 pre_mean_volume=None,
                 to_rgb=True,
                 size_divisor=None,
                 crop_size=None,
                 oversample=None,
                 random_crop=False,
                 resize_crop=False,
                 rescale_crop=False,
                 more_fix_crop=False,
                 multiscale_crop=False,
                 scales=None,
                 max_distort=1):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.pre_mean_volume = pre_mean_volume
        self.to_rgb = to_rgb
        self.size_divisor = size_divisor
        self.resize_crop = resize_crop
        self.rescale_crop = rescale_crop

        # croping parameters
        if crop_size is not None:
            if oversample == 'three_crop':
                self.op_crop = Group3CropSample(crop_size)
            elif oversample == 'ten_crop':
                # oversample crop (test)
                self.op_crop = GroupOverSample(crop_size)
            elif resize_crop:
                self.op_crop = RandomResizedCrop(crop_size)
            elif rescale_crop:
                self.op_crop = RandomRescaledCrop(crop_size)
            elif multiscale_crop:
                # multiscale crop (train)
                self.op_crop = GroupMultiScaleCrop(
                    crop_size, scales=scales, max_distort=max_distort,
                    fix_crop=not random_crop, more_fix_crop=more_fix_crop)
            else:
                # center crop (val)
                self.op_crop = GroupCenterCrop(crop_size)
        else:
            self.op_crop = None

    def __call__(self, img_group, scale, crop_history=None, flip=False,
                 keep_ratio=True, div_255=False, is_flow=False):

        if self.resize_crop or self.rescale_crop:
            img_group, crop_quadruple = self.op_crop(img_group)
            img_shape = img_group[0].shape
            scale_factor = None
        else:
            # 1. rescale
            if keep_ratio:
                tuple_list = [mmcv.imrescale(
                    img, scale, return_scale=True) for img in img_group]
                img_group, scale_factors = list(zip(*tuple_list))
                scale_factor = scale_factors[0]
            else:
                tuple_list = [mmcv.imresize(
                    img, scale, return_scale=True) for img in img_group]
                img_group, w_scales, h_scales = list(zip(*tuple_list))
                scale_factor = np.array([w_scales[0], h_scales[0],
                                         w_scales[0], h_scales[0]],
                                        dtype=np.float32)
            if self.pre_mean_volume is not None:
                volume_len = self.pre_mean_volume.shape[0]
                img_group = [img - self.pre_mean_volume[i % volume_len, ...]
                                 for i, img in enumerate(img_group)]
            # 2. crop (if necessary)
            if crop_history is not None:
                self.op_crop = GroupCrop(crop_history)
            if self.op_crop is not None:
                img_group, crop_quadruple = self.op_crop(
                    img_group, is_flow=is_flow)
            else:
                crop_quadruple = None

            img_shape = img_group[0].shape
        # 3. flip
        if flip:
            img_group = [mmcv.imflip(img) for img in img_group]
        if is_flow:
            for i in range(0, len(img_group), 2):
                img_group[i] = mmcv.iminvert(img_group[i])
        # 4a. div_255
        if div_255:
            img_group = [mmcv.imnormalize(img, 0, 255, False)
                         for img in img_group]
        # 4. normalize
        img_group = [mmcv.imnormalize(
            img, self.mean, self.std, self.to_rgb) for img in img_group]
        # 5. pad
        if self.size_divisor is not None:
            img_group = [mmcv.impad_to_multiple(
                img, self.size_divisor) for img in img_group]
            pad_shape = img_group[0].shape
        else:
            pad_shape = img_shape
        if is_flow:
            assert len(img_group[0].shape) == 2
            img_group = [np.stack((flow_x, flow_y), axis=2)
                         for flow_x, flow_y in zip(
                             img_group[0::2], img_group[1::2])]
        # 6. transpose
        img_group = [img.transpose(2, 0, 1) for img in img_group]

        # Stack into numpy.array
        img_group = np.stack(img_group, axis=0)
        return img_group, img_shape, pad_shape, scale_factor, crop_quadruple


class ImageTransform(object):
    """Preprocess an image.
    1. rescale the image to expected size
    2. normalize the image
    3. flip the image (if needed)
    4. pad the image (if needed)
    5. transpose to (c, h, w)
    """

    def __init__(self,
                 mean=(0, 0, 0),
                 std=(1, 1, 1),
                 to_rgb=True,
                 size_divisor=None):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb
        self.size_divisor = size_divisor

    def __call__(self, img, scale, flip=False, keep_ratio=True):
        if keep_ratio:
            img, scale_factor = mmcv.imrescale(img, scale, return_scale=True)
        else:
            img, w_scale, h_scale = mmcv.imresize(
                img, scale, return_scale=True)
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
        img_shape = img.shape
        img = mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
        if flip:
            img = mmcv.imflip(img)
        if self.size_divisor is not None:
            img = mmcv.impad_to_multiple(img, self.size_divisor)
            pad_shape = img.shape
        else:
            pad_shape = img_shape
        img = img.transpose(2, 0, 1)
        return img, img_shape, pad_shape, scale_factor


def bbox_flip(bboxes, img_shape):
    """Flip bboxes horizontally.
    Args:
        bboxes(ndarray): shape (..., 4*k)
        img_shape(tuple): (height, width)
    """
    assert bboxes.shape[-1] % 4 == 0
    w = img_shape[1]
    flipped = bboxes.copy()
    flipped[..., 0::4] = w - bboxes[..., 2::4] - 1
    flipped[..., 2::4] = w - bboxes[..., 0::4] - 1
    return flipped


def bbox_crop(bboxes, crop_quadruple):
    """Flip bboxes horizontally.
    Args:
        bboxes(ndarray): shape (..., 4*k)
        crop_quadruple(tuple): (x1, y1, tw, th)
    """
    assert bboxes.shape[-1] % 4 == 0
    assert crop_quadruple is not None
    cropped = bboxes.copy()
    x1, y1, tw, th = crop_quadruple
    cropped[..., 0::2] = bboxes[..., 0::2] - x1
    cropped[..., 1::2] = bboxes[..., 1::2] - y1
    return cropped


class BboxTransform(object):
    """Preprocess gt bboxes.
    1. rescale bboxes according to image size
    2. flip bboxes (if needed)
    3. pad the first dimension to `max_num_gts`
    """

    def __init__(self, max_num_gts=None):
        self.max_num_gts = max_num_gts

    def __call__(self, bboxes, img_shape, scale_factor, flip=False, crop=None):
        gt_bboxes = bboxes * scale_factor
        if crop is not None:
            gt_bboxes = bbox_crop(gt_bboxes, crop)
        if flip:
            gt_bboxes = bbox_flip(gt_bboxes, img_shape)
        gt_bboxes[:, 0::2] = np.clip(gt_bboxes[:, 0::2], 0, img_shape[1] - 1)
        gt_bboxes[:, 1::2] = np.clip(gt_bboxes[:, 1::2], 0, img_shape[0] - 1)
        if self.max_num_gts is None:
            return gt_bboxes
        else:
            num_gts = gt_bboxes.shape[0]
            padded_bboxes = np.zeros((self.max_num_gts, 4), dtype=np.float32)
            padded_bboxes[:num_gts, :] = gt_bboxes
            return padded_bboxes
