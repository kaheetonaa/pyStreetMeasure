import cv2
import torch

from MoGe.moge.model.v3 import MoGeModel

device = torch.device("cpu")

model = MoGeModel.from_pretrained("pretrained_model/moge3-vitl.pt").to(device)

input_image = cv2.cvtColor(cv2.imread("LC3D/img/DJI_20260730112036_0010.jpg"), cv2.COLOR_BGR2RGB)                       
input_image = torch.tensor(input_image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)    

output = model.infer(input_image)

print(output["depth"])
