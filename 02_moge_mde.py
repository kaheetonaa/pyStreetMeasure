import cv2
import torch

from MoGe.moge.model.v2 import MoGeModel

device = torch.device("cpu")

model = MoGeModel.from_pretrained("pretrained_model/moge-v2-vitl.pt").to(device)
print("done loading model!")
input_image = cv2.cvtColor(cv2.imread("LC3D/img/DJI_20260730112036_0010.jpg"), cv2.COLOR_BGR2RGB)                       
#resize_image
input_image=cv2.resize(input_image,(1000,1000), interpolation=cv2.INTER_AREA)
input_image = torch.tensor(input_image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)    
print("done processing image", input_image.shape)
output = model.infer(input_image)

print(output["depth"])
