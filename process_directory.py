"""
Applies style transfer to all the images in the directory.
uses the attention mask from the GLRC model.
NOTE: Needs "GLRC/python/" directory added to PYTHONPATH
"""

from __future__ import print_function
from pathlib import Path as P
import argparse

import torch

import process_stylization
from photo_wct import PhotoWCT

import attention_mask_api as mask_api

from PIL import Image

parser = argparse.ArgumentParser(description='Photorealistic Image Stylization')
parser.add_argument('--model', default='./PhotoWCTModels/photo_wct.pth',
                    help='Path to the PhotoWCT model. These are provided by the PhotoWCT submodule, please use `git submodule update --init --recursive` to pull.')
parser.add_argument('-c','--content_image_path', default='./images/content/')
parser.add_argument('--content_seg_path', default=[])
parser.add_argument('-s','--style_image_path', default='./images/style_with_labels/')
parser.add_argument('--style_seg_path', default=[])
parser.add_argument('--output_image_path', default='./results/')
parser.add_argument('--cuda', type=int, default=1, help='Enable CUDA.')
parser.add_argument('-glrc_1', '--glrc_model_1_path', default="/home/lperiyasa/GLRC/log/cpu_glrc_stage_1.pth.tar")
parser.add_argument('-glrc_2', '--glrc_model_2_path', default="/home/lperiyasa/GLRC/log/glrc_stage_2.pth.tar")
args = parser.parse_args()

# Load model
p_wct = PhotoWCT()
try:
    p_wct.load_state_dict(torch.load(args.model))
except:
    print("Fail to load PhotoWCT models. PhotoWCT submodule not updated?")
    exit()

print ('Done loading Model')
if args.cuda:
    p_wct.cuda(0)
    print ('WCT Model moved to GPU')


mask_extractor = mask_api.EXTRACT_ATTENTION_MASK(args.glrc_model_1_path, args.glrc_model_2_path)

content_path = P(args.content_image_path)
style_path   = P(args.style_image_path)
output_path  = P(args.output_image_path)

for i_content in content_path.glob('*.jpg'):
    print ('Processing ', str(i_content))
    for i_style in style_path.glob('*.jpg'):
        output_image_path = str (output_path / ( i_content.stem + '_' + i_style.stem + '.jpg' ))

        # extract and save the attention mask as label
        content_image =  Image.open(i_content)
        content_label = mask_extractor.extract(content_image)
        content_seg_path = str(i_content.parent / ( i_content.stem + '_' + 'label.png' ))
        content_label = Image.fromarray( content_label.astype('uint8'))
        content_label.save( content_seg_path )

        # constuct style_seg_path
        style_seg_path = str( i_style.parent / ( i_style.stem + '_' + 'label.png' ) )

        process_stylization.stylization(
            p_wct=p_wct,
            content_image_path=str(i_content),
            style_image_path=str(i_style),
            content_seg_path=content_seg_path,
            style_seg_path=style_seg_path,
            output_image_path=output_image_path,
            cuda=args.cuda,
        )
        print ('Done processing ', output_image_path)
print ('All images Done')

