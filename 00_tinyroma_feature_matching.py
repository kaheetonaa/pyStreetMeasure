from romatch import roma_outdoor
from PIL import Image
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np

device=torch.device("cpu")
roma_model = roma_outdoor(device=device)

img_A_path="LC3D/img/DJI_20260730113144_0092.jpg"
img_B_path="LC3D/img/1.jpeg"



img_A=cv2.imread(img_A_path)
img_B=cv2.imread(img_B_path)
#H_A, W_A, H_B, W_B=img_A.height,img_A.width,img_B.height,img_B.width
H_A, W_A, H_B, W_B=569,560,560,560
img_A_rs=cv2.resize(img_A,(560,560),interpolation=cv2.INTER_AREA)
img_B_rs=cv2.resize(img_B,(560,560),interpolation=cv2.INTER_AREA)
print("resized!")

matches,certainty=roma_model.match(Image.fromarray(img_A_rs),Image.fromarray(img_B_rs))
m_sample,c_sample=roma_model.sample(matches,certainty,num=100)
print(matches.shape)

kptsA, kptsB = roma_model.to_pixel_coordinates(m_sample, H_A, W_A, H_B, W_B)
print(kptsA.shape,kptsB.shape)

# Combine images side-by-side and draw match lines
canvas = np.hstack((img_A_rs, img_B_rs))
plt.figure(figsize=(12, 6))
plt.imshow(canvas)

for (x1, y1), (x2, y2) in zip(kptsA, kptsB):
    plt.plot([x1, x2 + W_A], [y1, y2], linewidth=0.8)

plt.axis("off")
plt.show()


H, mask = cv2.findHomography(
    kptsA.cpu().numpy(), kptsB.cpu().numpy(), ransacReprojThreshold=0.2, method=cv2.USAC_MAGSAC, confidence=0.999999, maxIters=10000
)

print(H)

pts = np.float32([[0, 0], [0, H_A], [W_A, H_A], [W_A, 0],[0,0]]).reshape(-1, 1, 2)

# applying perspective algorithm
dst = cv2.perspectiveTransform(pts, H)
print(dst)

_x = dst[:,0,0]
_y = dst[:,0,1]

plt.imshow(img_B_rs)
plt.plot(_x,_y,marker="o", linestyle="-", color="red", linewidth=2)
plt.show()

