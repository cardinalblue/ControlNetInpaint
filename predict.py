# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path, File

from diffusers import StableDiffusionInpaintPipeline, ControlNetModel, UniPCMultistepScheduler
from src.pipeline_stable_diffusion_controlnet_inpaint import *
from diffusers.utils import load_image

from PIL import Image
import numpy as np
import torch
from controlnet_aux import HEDdetector

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

        print(">>>> Predictor.setup")

        # Load control net
        print(">>>> Predictor.setup loading ControlNetModel")
        controlnet = ControlNetModel.from_pretrained(
            # "fusing/stable-diffusion-v1-5-controlnet-scribble",     # Load over network
            "./models/sd-controlnet-scribble",                      # Load from package
            torch_dtype=torch.float16
        )
        if torch.cuda.is_available():
            print(">>>> Predictor.setup using CUDA for ControlNetModel")
            controlnet.to('cuda')

        # Load inpainting pipeline
        print(">>>> Predictor.setup loading StableDiffusionControlNetInpaintPipeline")
        self.pipe_sd = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            # "runwayml/stable-diffusion-inpainting",     # Load over network
            "./models/stable-diffusion-inpainting",     # Load from package
            controlnet=controlnet, 
            torch_dtype=torch.float16
        )
        # speed up diffusion process with faster scheduler and memory optimization
        self.pipe_sd.scheduler = UniPCMultistepScheduler.from_config(self.pipe_sd.scheduler.config)

        if torch.cuda.is_available():
            print(">>>> Predictor.setup using CUDA for StableDiffusionControlNetInpaintPipeline")
            # remove following line if xformers is not installed
            self.pipe_sd.enable_xformers_memory_efficient_attention()
            self.pipe_sd.to('cuda')

        print(">>>> Predictor.setup finished")

    def predict(
        self,
        prompt:         str  = Input(description="Prompt text"),
        image:          Path = Input(description="Input image"),
        mask_image:     Path = Input(description="Mask image, white parts will be inpainted, black parts will be kept"),
        control_image:  Path = Input(description="Scribble image, black outline on white background"),
        num_outputs:    int  = Input(default=1, description="Number of images to generate per prompt"),
    ) -> List[Path]:
        """Run a single prediction on the model"""

        print(f">>>> Predictor.predict '{prompt}'")
        print(f">>>> Predictor.predict cuda: {torch.__version__} {torch.version.cuda} {self.pipe_sd}")

        # See https://github.com/andreasjansson/cog-stable-diffusion-inpainting/blob/master/predict.py for reference

        image                   = Image.open(image).convert("RGB")
        mask_image              = Image.open(mask_image).convert("RGB")
        control_image           = Image.open(control_image).convert("RGB")

        # ---- Generation!
        generator = torch.manual_seed(0)
        output = self.pipe_sd(
            prompt,
            num_inference_steps=20,
            generator=generator,
            image=image,
            control_image=control_image,
            mask_image=mask_image,
            num_images_per_prompt=num_outputs,
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