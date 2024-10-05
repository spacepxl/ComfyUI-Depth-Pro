# ComfyUI-Depth-Pro

Based on https://github.com/apple/ml-depth-pro

## Tips

The raw output of the depth model is metric depth (aka, distance from camera in meters) which may have values up in the hundreds or thousands for far away objects. This is great for projection to 3d, and you can use the focal length estimate to make a camera (focal_mm = focal_px * sensor_mm / sensor_px)

In order to convert metric depth to relative depth, like what's needed for controlnet, the depth has to be remapped into the 0 to 1 range, which is handled by a separate node. The defaults should be good for most uses, but you can invert it and/or use `gamma` to bias it brighter or darker.

If you get errors about "vit_large_patch14_dinov2" make sure timm is up to date (tested with 0.9.16 and 1.0.9)

## Example

![img](https://github.com/spacepxl/ComfyUI-Depth-Pro/blob/main/example/workflow.png)

## License

All code that is unique to this repository is covered by the Apache-2.0 license. Any 
code and models that are redistributed without modification from the original codebase 
may be subject to the original license from https://github.com/apple/ml-depth-pro/blob/main/LICENSE 
if applicable. This project is not affiliated in any way with Apple Inc.
