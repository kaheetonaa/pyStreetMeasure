import onnxruntime as ort
import onnx
import cv2
import numpy as np

def colorize_map(map, max_value=-1, invert=True):
    if max_value < 0:
        max_value = np.max(map)

    map = np.clip(map, 0, max_value)
    map = (map / max_value * 255).astype(np.uint8)
    if invert:
        map = 255 - map
    return map

path="midas_v21_384.onnx"

unidepth=ort.InferenceSession(path,providers=ort.get_available_providers())

image=cv2.imread("resized.jpg")

o_height,o_width,_=image.shape

def prepare_input(image):
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (384, 384)) #input requirements

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        input_img = (input_img / 255.0 - mean) / std
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

input_tensor=prepare_input(image)


outputs=unidepth.run(None,{"input_image": input_tensor})

out=outputs[0].transpose(1,2,0)
out=colorize_map(out)
out=cv2.resize(out,(o_width,o_height))
cv2.imwrite("out-resized-midas.jpg",out)

