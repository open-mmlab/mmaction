import argparse
import glob
import os.path as osp
from mmcv.lmdb.io import create_rawimage_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='generate lmdb datasets from raw frames')
    parser.add_argument(
        'root_dir', help='root directory to store the raw frames')
    parser.add_argument(
        'target_dir', help='target directory to stored the generated lmdbs')
    parser.add_argument('--image_format', nargs='+',
                        help='format of the images to be stored',
                        default=['img*.jpg'])
    parser.add_argument('--lmdb_tmpl', type=str,
                        help='template for the lmdb to be generated',
                        default='{}_img_lmdb')
    parser.add_argument('--image_tmpl', type=str,
                        help='template for the lmdb key', default=None)
    parser.add_argument('--modality', type=str, help='modality',
                        choices=['RGB', 'Flow'], default='RGB')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    video_path_list = glob.glob(osp.join(args.root_dir, '*'))
    for i, vv in enumerate(video_path_list):
        if not osp.isdir(vv):
            continue
        image_file_list = []
        for image_format in args.image_format:
            image_file_list += glob.glob(osp.join(vv, image_format))
        vid = vv.split('/')[-1]
        output_path = osp.join(args.target_dir, args.lmdb_tmpl.format(vid))
        create_rawimage_dataset(output_path, image_file_list,
                                image_tmpl=args.image_tmpl,
                                flag='color' if args.modality == 'RGB'
                                else 'grayscale',
                                check_valid=True)


if __name__ == '__main__':
    main()
