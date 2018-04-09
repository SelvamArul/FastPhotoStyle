"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from __future__ import print_function
from pathlib import Path as P
import argparse

import torch

import process_stylization
from photo_wct import PhotoWCT

parser = argparse.ArgumentParser(description='Photorealistic Image Stylization')
parser.add_argument('--model', default='./PhotoWCTModels/photo_wct.pth',
                    help='Path to the PhotoWCT model. These are provided by the PhotoWCT submodule, please use `git submodule update --init --recursive` to pull.')
parser.add_argument('-c','--content_image_path', default='./images/content/')
parser.add_argument('--content_seg_path', default=[])
parser.add_argument('-s','--style_image_path', default='./images/style/')
parser.add_argument('--style_seg_path', default=[])
parser.add_argument('--output_image_path', default='./results/')
parser.add_argument('--cuda', type=int, default=1, help='Enable CUDA.')
args = parser.parse_args()

# Load model
p_wct = PhotoWCT()
try:
    p_wct.load_state_dict(torch.load(args.model))
except:
    print("Fail to load PhotoWCT models. PhotoWCT submodule not updated?")
    exit()

if args.cuda:
    p_wct.cuda(0)

content_path = p(args.content_image_path)
style_path   = p(args.style_image_path)
output_path  = p(args.output_image_path)

for i_content in content_path.glob('*.jpg'):
    for i_style in style_path.glob('*.jpg'):
        output_image_path = str (output_path / ( i_content.stem + '_' + i_style.stem + '.jpg' ))
        process_stylization.stylization(
            p_wct=p_wct,
            content_image_path=str(i_content),
            style_image_path=str(i_style),
            content_seg_path=args.content_seg_path,
            style_seg_path=args.style_seg_path,
            output_image_path=output_image_path,
            cuda=args.cuda,
        )
    print ('Done processing ', output_image_path)
print ('All images Done')

