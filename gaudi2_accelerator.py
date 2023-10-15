import logging
import os
import sys

from pathlib import Path

import torch

from diffusers import DDIMScheduler, StableDiffusionPipeline
#from optimum.habana.diffusers import GaudiDDIMScheduler, GaudiStableDiffusionPipeline
#from optimum.habana.utils import set_seed

import modules.scripts as scripts

import gradio as gr
from PIL import Image, ImageOps

from modules import images
from modules.processing import process_images # Fast, default image processing
from modules.processing import StableDiffusionProcessing, Processed
# from modules import processing
# from modules.shared import opts, cmd_opts, state

def process_images_gaudi2(p: StableDiffusionProcessing):
    
    # """this is the main loop that both txt2img and img2img use; it calls func_init once inside all the scopes and func_sample once per batch"""
    # if (mode == 0 and p.enable_hr):
    #     print(p.hr_upscaler)

    try:
        import habana_frameworks.torch.core as htcore
        import habana_frameworks.torch.hpu as hthpu
    except:
        htcore = None
        hthpu = None

    if hthpu and hthpu.is_available():
        target = "hpu";
        print("Using HPU")
    elif torch.cuda.is_available():
        target = "cuda";
        print("Using GPU")
    else:
        target = "cpu"
        print("Using CPU")

    device = torch.device(target)

    model_name = "stabilityai/stable-diffusion-2-1-base"
    # model_name = "runwayml/stable-diffusion-v1-5" # Crashes
    # v1-5-pruned-emaonly
    scheduler = DDIMScheduler.from_pretrained(model_name, subfolder="scheduler")
    #scheduler = GaudiDDIMScheduler.from_pretrained(model_name, subfolder="scheduler")

    # infotexts = []
    # output_images = []

    # basename = ""
    # basename = Path(p.outpath_samples).stem

    #pipeline = GaudiStableDiffusionPipeline.from_pretrained(
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_name,
        scheduler=scheduler,
        #use_habana=True,
        #use_hpu_graphs=True,
        #gaudi_config="Habana/stable-diffusion-2",
        #torch_dtype=torch.bfloat16
        # torch_dtype="auto",
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    #    use_hpu_graphs_for_inference=True,
    #    use_lazy_mode=True,
    #    num_images_per_prompt=10,
    #    batch_size=2,

    pipeline.to("cuda")
    #pipeline.to("hpu")
    # pipeline.set_progress_bar_config(disable=True)
    # prompt = p.prompt
    # if prompt == "":
    #     prompt = "A beautiful painting of a singular lighthouse, shining its light across a sea of blood by greg rutkowski and thomas kinkade, Trending on artstation"
    # images = pipeline(prompt, num_inference_steps=50, guidance_scale=7.5, scheduler=scheduler, output_type="numpy")
    # images = images.images
    # for i in range(len(images)):
    #set_seed(p.seed)

    outputs = pipeline(
        num_images_per_prompt=10,
        #batch_size=2,
        prompt=p.prompt,
        negative_prompt=p.negative_prompt,
        num_inference_steps=p.steps,
        height=p.height,
        width=p.width,
        eta=p.eta,
        output_type="pil",
    )
        #styles=p.styles,
        #output_dir=p.outpath_samples,
        # Generate images
    #outputs = pipeline(
    #    prompt=args.prompts,
    #    num_images_per_prompt=args.num_images_per_prompt,
    #    batch_size=args.batch_size,
    #    height=args.height,
    #    width=args.width,
    #    num_inference_steps=args.num_inference_steps,
    #    guidance_scale=args.guidance_scale,
    #    negative_prompt=args.negative_prompts,
    #    eta=args.eta,
    #    output_type=args.output_type,
    #)
    for i, image in enumerate(outputs.images):
        image.save(f"image_{i+1}.png")

    #image = Image.fromarray(outputs)
    #images.save_image(image, p.outpath_samples, "", p.seeds[i], p.prompts[i], opts.samples_format, info=infotext(n, i), p=p)

    #text = infotext(n, i)
    #infotexts.append(text)
    #image.info["parameters"] = text
    #output_images.append(image)

    #result = Processed(
    #    p,
    #    images_list=output_images,
    #    seed=p.all_seeds[0],
    #    info=infotext(),
    #    comments="".join(f"{comment}\n" for comment in comments),
    #    subseed=p.all_subseeds[0],
    #    index_of_first_image=index_of_first_image,
    #    infotexts=infotexts,
    #)

    return outputs

class Script(scripts.Script):  

# The title of the script. This is what will be displayed in the dropdown menu.
    def title(self):

        return "Gaudi2 Accelerator"


# Determines when the script should be shown in the dropdown menu via the 
# returned value. As an example:
# is_img2img is True if the current tab is img2img, and False if it is txt2img.
# Thus, return is_img2img to only show the script on the img2img tab.

    def show(self, is_img2img):
        # self.is_img2img = False

        return True

# How the script's is displayed in the UI. See https://gradio.app/docs/#components
# for the different UI components you can use and how to create them.
# Most UI components can return a value, such as a boolean for a checkbox.
# The returned values are passed to the run method as parameters.

    def ui(self, is_img2img):
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

    def run(self, p):

        proc = process_images_gaudi2(p)
        # proc = process_images(p)
        # basename = "image"

        # use the save_images method from images.py to save
        #for i in range(len(processed.images)):
        #    images.save_image(image=processed.images[i], path="outputs", basename=basename, prompt=p.prompt, p=p)
            # save_image(image, path, basename, seed=None, prompt=None, extension='png', info=None, short_filename=False, no_prompt=False, grid=False, pnginfo_section_name='parameters', p=None, existing_info=None, forced_filename=None, suffix="", save_to_dirs=None):


        return proc
