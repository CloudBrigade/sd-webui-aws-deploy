import logging
import os
import sys

from pathlib import Path

import torch

from optimum.habana.diffusers import GaudiDDIMScheduler, GaudiStableDiffusionPipeline
from optimum.habana.utils import set_seed

import modules.scripts as scripts
import gradio as gr
import os

# from modules import images
# from modules.processing import process_images, Processed
from modules.processing import StableDiffusionProcessing, Processed
from modules.shared import opts, cmd_opts, state

def process_images_gaudi2(p: StableDiffusionProcessing):
    
    # """this is the main loop that both txt2img and img2img use; it calls func_init once inside all the scopes and func_sample once per batch"""
    # if (mode == 0 and p.enable_hr):
    #     print(p.hr_upscaler)

    model_name = "stabilityai/stable-diffusion-2-1"
    scheduler = GaudiDDIMScheduler.from_pretrained(model_name, subfolder="scheduler")

    # basename = Path(p.outpath_samples).stem

    pipeline = GaudiStableDiffusionPipeline.from_pretrained(
        model_name,
        scheduler=scheduler,
        prompt=p.prompt,
        negative_prompt=p.negative_prompt,
        num_inference_steps=p.steps,
        height=p.height,
        width=p.width,
        seed=p.seed,
        eta=p.eta,
        output_dir=p.outpath_samples,
        use_habana=True,
        use_hpu_graphs=True,
        gaudi_config="Habana/stable-diffusion-2",
        torch_dtype=torch.bfloat16
    )

    # pipeline.to("hpu")
    # pipeline.set_progress_bar_config(disable=True)
    # set_seed(p.seed)
    # prompt = p.prompt
    # if prompt == "":
    #     prompt = "A beautiful painting of a singular lighthouse, shining its light across a sea of blood by greg rutkowski and thomas kinkade, Trending on artstation"
    # images = pipeline(prompt, num_inference_steps=50, guidance_scale=7.5, scheduler=scheduler, output_type="numpy")
    # images = images.images
    # for i in range(len(images)):

    # outputs = pipeline(
    #     ["An image of a squirrel in Picasso style"],
    #     num_images_per_prompt=10,
    #     batch_size=2,
    #     height=512,
    #     width=512,
    # )

    return

class Script(scripts.Script):  

# The title of the script. This is what will be displayed in the dropdown menu.
    def title(self):

        return "Gaudi2 Accelerator"


# Determines when the script should be shown in the dropdown menu via the 
# returned value. As an example:
# is_img2img is True if the current tab is img2img, and False if it is txt2img.
# Thus, return is_img2img to only show the script on the img2img tab.

    def show(self, is_txt2img):

        return is_txt2img

# How the script's is displayed in the UI. See https://gradio.app/docs/#components
# for the different UI components you can use and how to create them.
# Most UI components can return a value, such as a boolean for a checkbox.
# The returned values are passed to the run method as parameters.

    def ui(self, is_txt2img):
        # angle = gr.Slider(minimum=0.0, maximum=360.0, step=1, value=0,
        # label="Angle")
        # hflip = gr.Checkbox(False, label="Horizontal flip")
        # vflip = gr.Checkbox(False, label="Vertical flip")
        # overwrite = gr.Checkbox(False, label="Overwrite existing files")
        # return [angle, hflip, vflip, overwrite]
        return


# This is where the additional processing is implemented. The parameters include
# self, the model object "p" (a StableDiffusionProcessing class, see
# processing.py), and the parameters returned by the ui method.
# Custom functions can be defined here, and additional libraries can be imported 
# to be used in processing. The return value should be a Processed object, which is
# what is returned by the process_images method.

    def run(self, p, angle, hflip, vflip, overwrite):

        processed = process_images_gaudi2(p)

        # # rotate and flip each image in the processed images
        # # use the save_images method from images.py to save
        # # them.
        # for i in range(len(proc.images)):

        #     proc.images[i] = rotate_and_flip(proc.images[i], angle, hflip, vflip)

        #     images.save_image(proc.images[i], p.outpath_samples, basename,
        #     proc.seed + i, proc.prompt, opts.samples_format, info= proc.info, p=p)

        return processed