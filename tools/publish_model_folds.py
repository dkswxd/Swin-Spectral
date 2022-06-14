# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import subprocess

import torch
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description='Process a checkpoint to be published')
    parser.add_argument('in_file', help='input checkpoint filename')
    parser.add_argument('out_file', help='output checkpoint filename')
    parser.add_argument('folds', type=int, default=5)
    args = parser.parse_args()
    return args


def process_checkpoint(in_file, out_file):
    checkpoint = torch.load(in_file, map_location='cpu')
    # remove optimizer for smaller file size
    if 'optimizer' in checkpoint:
        del checkpoint['optimizer']
    if 'meta' in checkpoint:
        keys = list(checkpoint['meta'].keys())
        for key in keys:
            if not (key == 'CLASSES' or key == 'PALETTE'):
                del checkpoint['meta'][key]
    torch.nn.init.trunc_normal_(checkpoint['state_dict']['auxiliary_head.convs.0.bn.bias'],std=0.002)
    # if it is necessary to remove some sensitive data in checkpoint['meta'],
    # add the code here.
    os.mkdir(out_file.rsplit('/',1)[0])
    torch.save(checkpoint, out_file)
    # sha = subprocess.check_output(['sha256sum', out_file]).decode()
    # final_file = out_file.rstrip('.pth') + '-{}.pth'.format(sha[:8])
    # subprocess.Popen(['mv', out_file, final_file])


def main():
    args = parse_args()
    for fold in range(args.folds):
        process_checkpoint(args.in_file.format(fold), args.out_file.format(fold))


if __name__ == '__main__':
    main()
