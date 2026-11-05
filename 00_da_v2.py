import cv2
import torch
from DA2.depth_anything_v2.dpt import DepthAnythingV2
import matplotlib.pyplot as plt
import numpy as np
import argparse

#---parameters---------------------------------------------------
parser = argparse.ArgumentParser(description ='Running Depth Anything_v2 model on photo, output as npy data',add_help=True)
parser.add_argument("--input",help="input photo path",type=str,required=True)
parser.add_argument("--output",help="output npy path",type=str,required=True)
parser.add_argument("--device",help="device for the inference:cuda, cpu,etc.",type=str,default="cpu")
parser.add_argument("--model",help="model path",type=str,default="'DA2/depth_anything_v2_vitl.pth",required=True)
parser.add_argument("--resize",help="resize scale image. e.g. 0.1 for rescale 10 times smaller",type=float,default=0.1)

#input, output, device, model, resize

args = parser.parse_args()

#---variables----------------------------------------------------
DEVICE = 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

#---load model---------------------------------------------------
model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(args.model, map_location='cpu'))
model = model.to(DEVICE).eval()

#---load image---------------------------------------------------
raw_img = cv2.imread(args.input)
_scale=args.resize
_h,_w=int(raw_img.shape[0]*_scale),int(raw_img.shape[1]*_scale)
raw_img=cv2.resize(raw_img,(_w,_h),interpolation=cv2.INTER_LINEAR)
#----------------------------------------------------------------
# HxW raw depth map in numpy
depth = model.infer_image(raw_img) 
np.save(args.output,depth)
plt.imshow(depth)
plt.show()
