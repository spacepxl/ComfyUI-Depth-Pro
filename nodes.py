import os
import numpy as np
import torch
from tqdm import trange
from torchvision.transforms import Normalize

import comfy.utils
import model_management 
import folder_paths

from . import depth_pro


class LoadDepthPro:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "precision": (["fp16", "fp32"],),
                },
            }
    
    RETURN_TYPES = ("DEPTH_PRO_MODEL",)
    RETURN_NAMES = ("depth_pro_model",)
    FUNCTION = "load_model"
    CATEGORY = "Depth-Pro"
    
    def load_model(self, precision):
        device = model_management.get_torch_device()
        dtype = torch.float16 if precision == "fp16" else torch.float32
        
        depth_model_path = os.path.join(folder_paths.models_dir, "depth", "ml-depth-pro")
        if not os.path.exists(depth_model_path):
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="spacepxl/ml-depth-pro",
                local_dir=depth_model_path,
                local_dir_use_symlinks=False,
                )
        
        depth_model_path = os.path.join(depth_model_path, "depth_pro.fp16.safetensors")
        model, transform = depth_pro.create_model_and_transforms(depth_model_path, device=device, precision=dtype)
        model.eval()
        
        model_dict = {
            "model": model,
            "device": device,
            "dtype": dtype,
            }
        
        return (model_dict,)


class DepthPro:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "depth_pro_model": ("DEPTH_PRO_MODEL",),
                "image": ("IMAGE",),
                },
            }
    
    RETURN_TYPES = ("IMAGE", "LIST", "FLOAT",)
    RETURN_NAMES = ("metric_depth", "focal_list", "focal_avg",)
    FUNCTION = "estimate_depth"
    CATEGORY = "Depth-Pro"
    
    def estimate_depth(self, depth_pro_model, image):
        model = depth_pro_model["model"]
        device = depth_pro_model["device"]
        dtype = depth_pro_model["dtype"]
        
        transform = Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        
        rgb = image.unsqueeze(0) if len(image.shape) < 4 else image
        rgb = rgb.movedim(-1, 1) # BCHW
        
        depth = []
        focal_px = []
        
        # add comfyui progress bar
        for i in trange(rgb.size(0)):
            rgb_image = rgb[i, :3].unsqueeze(0).to(device, dtype=dtype)
            rgb_image = transform(rgb_image)
            
            prediction = model.infer(rgb_image)
            depth.append(prediction["depth"].unsqueeze(-1))
            focal_px.append(prediction["focallength_px"].item())
        
        depth = torch.stack(depth, dim=0).repeat(1,1,1,3)
        focal_list = focal_px
        focal_avg = np.mean(focal_px)
        
        return (depth, focal_list, focal_avg)


class MetricDepthToRelative:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "depth": ("IMAGE",),
                "per_image": ("BOOLEAN", {"default": False,}),
                "invert": ("BOOLEAN", {"default": True,}),
                "std_dev": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 100.0, "step": 0.1,  "round": 0.1}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100, "step": 0.01}),
                },
            }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depth",)
    FUNCTION = "convert_depth"
    CATEGORY = "Depth-Pro"
    
    def convert_depth(self, depth, per_image, invert, std_dev, gamma):
        relative_depth = depth.detach().clone()
        
        if per_image:
            for i in range(relative_depth.size(0)):
                std, mean = torch.std_mean(relative_depth[i], dim=None)
                relative_depth[i] = torch.clamp(relative_depth[i], min = 0, max = std * std_dev + mean)
                relative_depth[i] = relative_depth[i] - relative_depth[i].min()
                relative_depth[i] = relative_depth[i] / relative_depth[i].max()
        else:
            std, mean = torch.std_mean(relative_depth, dim=None)
            relative_depth = torch.clamp(relative_depth, min = 0, max = std * std_dev + mean)
            relative_depth = relative_depth - relative_depth.min()
            relative_depth = relative_depth / relative_depth.max()
        
        if invert:
            relative_depth = 1 - relative_depth
        
        if gamma != 1:
            relative_depth = relative_depth ** (1 / gamma)
        
        return (relative_depth,)


NODE_CLASS_MAPPINGS = {
    "LoadDepthPro": LoadDepthPro,
    "DepthPro": DepthPro,
    "MetricDepthToRelative": MetricDepthToRelative,
    }

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadDepthPro": "(Down)Load Depth Pro model",
    "DepthPro": "Depth Pro",
    "MetricDepthToRelative": "Metric Depth to Relative",
    }