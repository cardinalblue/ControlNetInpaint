# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path, File

from diffusers import StableDiffusionInpaintPipeline, ControlNetModel, UniPCMultistepScheduler
from src.pipeline_stable_diffusion_controlnet_inpaint import *
from diffusers.utils import load_image

import cv2
from PIL import Image
import numpy as np
import torch
from matplotlib import pyplot as plt
from controlnet_aux import HEDdetector

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

        # Load HED (this model is small, get from HuggingFace)
        self.hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')

        # Load control net
        controlnet = ControlNetModel.from_pretrained(
            "./models/sd-controlnet-scribble", torch_dtype=torch.float16
        )

        # Load inpainting pipeline
        self.pipe_sd = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "./models/stable-diffusion-inpainting", controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None
        )
        # speed up diffusion process with faster scheduler and memory optimization
        self.pipe_sd.scheduler = UniPCMultistepScheduler.from_config(self.pipe_sd.scheduler.config)

        # remove following line if xformers is not installed
        self.pipe_sd.enable_xformers_memory_efficient_attention()

        self.pipe_sd.to('cuda')

    def predict(
        self,
        prompt:         str  = Input(description="Prompt text"),
        image:          Path = Input(description="Input image"),
        mask_image:     Path = Input(description="Mask image"),
        scribble_image: Path = Input(description="Scribble image"),
    ) -> Path:
        """Run a single prediction on the model"""

        # See https://github.com/andreasjansson/cog-stable-diffusion-inpainting/blob/master/predict.py for reference

        image           = Image.open(image).convert("RGB")
        mask_image      = Image.open(mask_image).convert("RGB")
        scribble_image  = Image.open(scribble_image).convert("RGB")

        control_image = self.hed(image, scribble=True)

        # ---- Generation!
        generator = torch.manual_seed(0)
        output = self.pipe_sd(
            prompt,
            num_inference_steps=20,
            generator=generator,
            image=image,
            control_image=control_image,
            mask_image=mask_image
        )

        # ---- Process output
        samples = [
            output.images[i]
            for i, nsfw_flag in enumerate(output.nsfw_content_detected)
            if not nsfw_flag
        ]
        if len(samples) == 0:
            raise Exception(
                f"NSFW content detected. Try running it again, or try a different prompt."
            )
        output_paths = []
        for i, sample in enumerate(samples):
            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths