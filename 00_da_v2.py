import cv2
import torch
import sys
sys.path.insert(0, "Depth-Anything-V2/depth_anything_v2/dpt/ ")
import DepthAnythingV2

DEVICE = 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'Depth-Anything-V2/depth_anything_v2_vitl.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

raw_img = cv2.imread('LC3D/img/DJI_20260730113144_0092.jpg')
print(raw_img.shape)
#raw_img=cv2.resize(raw_img,(780, 540), interpolation=cv2.INTER_LINEAR)
#depth = model.infer_image(raw_img) # HxW raw depth map in numpy
