import time
from typing import Optional, Sequence
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from PIL import Image

from depth_anything_3.cfg import create_object, load_config
from depth_anything_3.registry import MODEL_REGISTRY
from depth_anything_3.specs import Prediction
from depth_anything_3.utils.export import export
from depth_anything_3.utils.geometry import affine_inverse
from depth_anything_3.utils.io.input_processor import InputProcessor
from depth_anything_3.utils.io.output_processor import OutputProcessor
from depth_anything_3.utils.logger import logger
from depth_anything_3.utils.pose_align import align_poses_umeyama

torch.backends.cudnn.benchmark = False

class DepthAnything3(nn.Module, PyTorchModelHubMixin):
    """
    CPU-only Depth Anything 3 API class.
    This class provides depth estimation with all tensors and models on CPU.
    """

    _commit_hash: str | None = None  # Set by mixin when loading from Hub

    def __init__(self, model_name: str = "da3-large", **kwargs):
        super().__init__()
        self.model_name = model_name

        # Build network and force CPU
        self.config = load_config(MODEL_REGISTRY[self.model_name])
        self.model = create_object(self.config)
        self.model.eval()
        self.device = torch.device("cpu")
        self.model.to(self.device)

        # Initialize processors
        self.input_processor = InputProcessor()
        self.output_processor = OutputProcessor()

    @torch.inference_mode()
    def forward(
        self,
        image: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        export_feat_layers: list[int] | None = None,
        infer_gs: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Forward pass on CPU."""
        image = image.to(self.device)
        return self.model(image, extrinsics, intrinsics, export_feat_layers, infer_gs)

    def inference(
        self,
        image: list[np.ndarray | Image.Image | str],
        extrinsics: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
        align_to_input_ext_scale: bool = True,
        infer_gs: bool = False,
        render_exts: np.ndarray | None = None,
        render_ixts: np.ndarray | None = None,
        render_hw: tuple[int, int] | None = None,
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
        export_dir: str | None = None,
        export_format: str = "mini_npz",
        export_feat_layers: Sequence[int] | None = None,
        conf_thresh_percentile: float = 40.0,
        num_max_points: int = 1_000_000,
        show_cameras: bool = True,
        feat_vis_fps: int = 15,
        export_kwargs: Optional[dict] = {},
    ) -> Prediction:
        """Run inference on input images (CPU)."""
        if "gs" in export_format:
            assert infer_gs, "must set `infer_gs=True` to perform gs-related export."

        imgs_cpu, extrinsics, intrinsics = self._preprocess_inputs(
            image, extrinsics, intrinsics, process_res, process_res_method
        )
        imgs, ex_t, in_t = self._prepare_model_inputs(imgs_cpu, extrinsics, intrinsics)
        ex_t_norm = self._normalize_extrinsics(ex_t.clone() if ex_t is not None else None)
        export_feat_layers = list(export_feat_layers) if export_feat_layers is not None else []

        raw_output = self._run_model_forward(imgs, ex_t_norm, in_t, export_feat_layers, infer_gs)
        prediction = self._convert_to_prediction(raw_output)
        prediction = self._align_to_input_extrinsics_intrinsics(extrinsics, intrinsics, prediction, align_to_input_ext_scale)
        prediction = self._add_processed_images(prediction, imgs_cpu)

        if export_dir is not None:
            if "gs" in export_format and infer_gs:
                if "gs_video" not in export_format:
                    export_format = f"{export_format}-gs_video"
                if "gs_video" in export_format and "gs_video" not in export_kwargs:
                    export_kwargs["gs_video"] = {}
                    export_kwargs["gs_video"].update({
                        "extrinsics": render_exts,
                        "intrinsics": render_ixts,
                        "out_image_hw": render_hw,
                    })
            if "glb" in export_format:
                if "glb" not in export_kwargs:
                    export_kwargs["glb"] = {}
                export_kwargs["glb"].update({
                    "conf_thresh_percentile": conf_thresh_percentile,
                    "num_max_points": num_max_points,
                    "show_cameras": show_cameras,
                })
            if "feat_vis" in export_format:
                if "feat_vis" not in export_kwargs:
                    export_kwargs["feat_vis"] = {}
                export_kwargs["feat_vis"].update({"fps": feat_vis_fps})
            self._export_results(prediction, export_format, export_dir, **export_kwargs)

        return prediction

    def _preprocess_inputs(self, image, extrinsics=None, intrinsics=None, process_res=504, process_res_method="upper_bound_resize"):
        """Preprocess input images on CPU."""
        start_time = time.time()
        imgs_cpu, extrinsics, intrinsics = self.input_processor(
            image,
            extrinsics.copy() if extrinsics is not None else None,
            intrinsics.copy() if intrinsics is not None else None,
            process_res,
            process_res_method,
        )
        logger.info("Processed Images Done taking", time.time() - start_time, "seconds. Shape:", imgs_cpu.shape)
        return imgs_cpu, extrinsics, intrinsics

    def _prepare_model_inputs(self, imgs_cpu, extrinsics, intrinsics):
        """Prepare tensors for model input (CPU-only)."""
        imgs = imgs_cpu[None].float().to(self.device)
        ex_t = extrinsics[None].float().to(self.device) if extrinsics is not None else None
        in_t = intrinsics[None].float().to(self.device) if intrinsics is not None else None
        return imgs, ex_t, in_t

    def _normalize_extrinsics(self, ex_t):
        if ex_t is None:
            return None
        transform = affine_inverse(ex_t[:, :1])
        ex_t_norm = ex_t @ transform
        c2ws = affine_inverse(ex_t_norm)
        translations = c2ws[..., :3, 3]
        dists = translations.norm(dim=-1)
        median_dist = torch.clamp(torch.median(dists), min=1e-1)
        ex_t_norm[..., :3, 3] = ex_t_norm[..., :3, 3] / median_dist
        return ex_t_norm

    def _align_to_input_extrinsics_intrinsics(self, extrinsics, intrinsics, prediction, align_to_input_ext_scale=True, ransac_view_thresh=10):
        if extrinsics is None:
            return prediction
        prediction.intrinsics = intrinsics.numpy()
        _, _, scale, aligned_extrinsics = align_poses_umeyama(
            prediction.extrinsics,
            extrinsics.numpy(),
            ransac=len(extrinsics) >= ransac_view_thresh,
            return_aligned=True,
            random_state=42,
        )
        if align_to_input_ext_scale:
            prediction.extrinsics = extrinsics[..., :3, :].numpy()
            prediction.depth /= scale
        else:
            prediction.extrinsics = aligned_extrinsics
        return prediction

    def _run_model_forward(self, imgs, ex_t, in_t, export_feat_layers=None, infer_gs=False):
        start_time = time.time()
        output = self.forward(imgs, ex_t, in_t, export_feat_layers, infer_gs)
        logger.info(f"Model Forward Pass Done (CPU). Time: {time.time() - start_time} seconds")
        return output

    def _convert_to_prediction(self, raw_output):
        start_time = time.time()
        output = self.output_processor(raw_output)
        logger.info(f"Conversion to Prediction Done. Time: {time.time() - start_time} seconds")
        return output

    def _add_processed_images(self, prediction, imgs_cpu):
        processed_imgs = imgs_cpu.permute(0, 2, 3, 1).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        processed_imgs = np.clip(processed_imgs * std + mean, 0, 1)
        processed_imgs = (processed_imgs * 255).astype(np.uint8)
        prediction.processed_images = processed_imgs
        return prediction

    def _export_results(self, prediction, export_format, export_dir, **kwargs):
        start_time = time.time()
        export(prediction, export_format, export_dir, **kwargs)
        logger.info(f"Export Results Done. Time: {time.time() - start_time} seconds")

    def _get_model_device(self):
        """Always return CPU device."""
        return torch.device("cpu")
