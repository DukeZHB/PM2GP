"""Stage 1: pseudo-mask generation with a standard CAM pipeline.

This package is a *supporting* component of PM2GP. It follows the
conventional CAM-based weakly supervised pipeline (image-level
multi-label classifier -> class activation maps -> pseudo masks) and is
NOT a contribution of the paper. The contributions of PM2GP are the
diffusion-based modules in ``diffusion/`` and ``racd/`` and the
segmentation training in ``segmentation/``.
"""
