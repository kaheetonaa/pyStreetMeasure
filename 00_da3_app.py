# app.py (safe CPU startup for HF Spaces)
import os
import io
import numpy as np
import torch
from PIL import Image,ImageDraw
import gradio as gr

# Import the CPU-patched class you added earlier
from depth_anything_3.api import DepthAnything3

# ---------------------------
# Configuration
# ---------------------------
# Keep the same model path you used earlier (default is the one in your logs)
MODEL_DIR = os.environ.get("DA3_MODEL_DIR", "depth-anything/DA3NESTED-GIANT-LARGE")

# Lower processing resolution to make CPU inference feasible.
# Increase if you want better quality but expect it to be much slower.
PROCESS_RES = int(os.environ.get("DA3_PROCESS_RES", "384"))

# ---------------------------
# Model loading (CPU)
# ---------------------------
print(f"🔄 Loading DepthAnything3 from '{MODEL_DIR}' on CPU (this may take a moment)...")
# Uses the PyTorchModelHubMixin.from_pretrained you have in the class
model = DepthAnything3.from_pretrained(MODEL_DIR)
model.to(torch.device("cpu"))
model.eval()
print("✅ Model ready on CPU")

# ---------------------------
# Inference helper
# ---------------------------
def _normalize_depth_to_uint8(depth: np.ndarray) -> np.ndarray:
    """Normalize a depth map (H,W) to uint8 grayscale for display."""
    if depth is None:
        return None
    # convert to float
    d = depth.astype(np.float32)
    # clip NaNs / infs
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    # Normalize robustly: use 1st and 99th percentiles to avoid outliers
    vmin = np.percentile(d, 1.0)
    vmax = np.percentile(d, 99.0)
    if vmax - vmin < 1e-6:
        vmax = vmin + 1.0
    d = (d - vmin) / (vmax - vmin)
    d = np.clip(d, 0.0, 1.0)
    img = (d * 255.0).astype(np.uint8)
    return img

def depth_measure(depth: np.ndarray,coord: np.ndarray) -> float:
    if depth is None:
        return None
    if coord is None:
        return None
    return depth[coord[0]][coord[1]]
    

def run_depth(single_img: Image.Image, process_res: int = PROCESS_RES, x_coord:int=0, y_coord:int =0):
    """
    Run single-image depth inference with the patched DepthAnything3 API.
    Returns a grayscale PIL image visualizing depth.
    """
    if single_img is None:
        return None

    # Convert PIL to numpy (DepthAnything3 accepts PIL images)
    try:
        # Use the API's inference function; we pass a list with single image.
        # Keep other args minimal to avoid heavy processing.
        pred = model.inference(
            [single_img],
            process_res=process_res,
            process_res_method="upper_bound_resize",
            export_dir="da3-output",
            export_format="mini_npz-glb",
        )
    except Exception as e:
        # If inference raises, return a helpful message image
        msg = f"Inference error: {e}"
        print(msg)
        # Make a small image with the error text
        err_img = Image.new("RGB", (640, 120), color=(255, 255, 255))
        return err_img

    # Extract depth from Prediction object - handle a few possible shapes / attrs
    depth_map = None
    # First try attribute .depth (common pattern in your code)
    if hasattr(pred, "depth"):
        depth_map = pred.depth
    elif isinstance(pred, dict) and "depth" in pred:
        depth_map = pred["depth"]
    elif hasattr(pred, "predictions") and len(pred.predictions) > 0:
        # fallback: some wrappers store lists
        depth_map = pred.predictions[0].depth if hasattr(pred.predictions[0], "depth") else None

    # depth_map might be (N,H,W) or (H,W)
    if depth_map is None:
        # fallback: try processed_images if available (visual sanity)
        try:
            if hasattr(pred, "processed_images"):
                imgs = pred.processed_images
                if isinstance(imgs, np.ndarray) and imgs.shape[0] > 0:
                    # return first processed image
                    return Image.fromarray((imgs[0] * 255).astype(np.uint8))
        except Exception:
            pass
        # nothing usable
        print("No depth found in prediction; returning empty image.")
        return Image.new("RGB", (640, 480), color=(255, 255, 255))

    # If depth_map is batched, take first
    if isinstance(depth_map, (list, tuple)):
        depth_map = depth_map[0]
    if isinstance(depth_map, np.ndarray) and depth_map.ndim == 3 and depth_map.shape[0] in (1,):
        # shape (1,H,W)
        depth_map = depth_map[0]
    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.cpu().numpy()
    #print(depth_map)
    #coord=np.array([0,0])
    #coord[0]=y_coord/single_img.size[1]*depth_map.shape[0]
    #coord[1]=x_coord/single_img.size[0]*depth_map.shape[1]
    #print('measure:', depth_measure(depth_map,coord))

    # Now depth_map should be (H,W)
    if depth_map.ndim == 3 and depth_map.shape[0] == 3:
        # if somehow 3-channel, convert to single channel by averaging
        depth_map = depth_map.mean(axis=0)

    depth_uint8 = _normalize_depth_to_uint8(depth_map)
    if depth_uint8 is None:
        return Image.new("RGB", (640, 480), color=(255, 255, 255))
    # Return grayscale PIL image
    depth_img = Image.fromarray(depth_uint8, mode="L")#.convert('RGB')
    #annotation=ImageDraw.Draw(depth_img)
    #annotation.point([coord[1],coord[0]],fill=(255,0,0))
    return depth_img

# ---------------------------
# Gradio interface
# ---------------------------
title = "📏StreetMeasure"
description = (
    "a fork of Depth Anything 3 — CPU (single-image)\n"
    "This application estimate the depth between two points on the photo and the physical distance between them."
)

# Make the Gradio Interface the top-level `app` variable so HF Spaces detects it
app = gr.Interface(
    fn=run_depth,
    inputs=[
        gr.Image(type="pil", label="Upload image"),
        gr.Slider(minimum=128, maximum=1024, step=64, value=PROCESS_RES, label="Process resolution (smaller = faster)")],
       # gr.Number(label='x_coord'),
       # gr.Number(label='y_coord')],
    outputs=gr.Image(label="Predicted depth (grayscale)"),
    title=title,
    description=description,
)

# For local running
if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
