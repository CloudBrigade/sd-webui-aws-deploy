import os
import sys
import logging
from pathlib import Path

import gradio as gr
from PIL import Image, ImageOps

import torch

import modules.scripts as scripts
from modules import images, devices
from modules.processing import process_images  # Fast, default image processing
from modules.processing import StableDiffusionProcessing, Processed, create_infotext
from modules.shared import state

image_save_path = Path("outputs/txt2img-images")
os.makedirs(image_save_path, exist_ok=True)


def process_images_gaudi2(p: StableDiffusionProcessing) -> Processed:
    # Verify hardward and select HPU or GPU if available, in that order
    try:
        import habana_frameworks.torch.core as htcore
        import habana_frameworks.torch.hpu as hthpu
    except:
        htcore = None
        hthpu = None

    if hthpu and hthpu.is_available():
        target = "hpu"
        print("Using HPU")
    elif torch.cuda.is_available():
        target = "cuda"
        print("Using GPU")
    else:
        target = "cpu"
        print("Using CPU")

    device = torch.device(target)
    print(device)

    # model_name = "stabilityai/stable-diffusion-2-1-base"
    # model_name = "./v1-5-pruned-emaonly"
    model_name = "runwayml/stable-diffusion-v1-5" # Crashes

    if target == "hpu":
        print("Using Gaudi Pipeline")

        from optimum.habana.diffusers import (
            GaudiDDIMScheduler,
            GaudiStableDiffusionPipeline,
        )
        from optimum.habana.utils import set_seed

        scheduler = GaudiDDIMScheduler.from_pretrained(
            model_name, subfolder="scheduler"
        )
        # pipeline = GaudiStableDiffusionPipeline.from_single_file(
        pipeline = GaudiStableDiffusionPipeline.from_pretrained(
            model_name,
            scheduler=scheduler,
            use_habana=True,
            use_hpu_graphs=True,
            gaudi_config="Habana/stable-diffusion-2",
            device_map="auto",
        )
        #    use_hpu_graphs_for_inference=True,
        #    use_lazy_mode=True,
        set_seed(p.seed)
    else:
        print("Using SD Pipeline")
        from diffusers import (
            DDIMScheduler,
            StableDiffusionPipeline,
        )

        scheduler = DDIMScheduler.from_pretrained(
            model_name, subfolder="scheduler"
        )
        # pipeline = StableDiffusionPipeline.from_pretrained(
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            scheduler=scheduler,
            device_map="auto",
            # torch_dtype=torch.bfloat16,
            # torch_dtype=torch.float32,
            # torch_dtype=auto, # errors out
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        #    use_lazy_mode=True,

    pipeline.to(target)
    # pipeline.set_progress_bar_config(disable=True)

    def infotext(iteration=0, position_in_batch=0):
        return create_infotext(
            p,
            p.all_prompts,
            p.all_seeds,
            p.all_subseeds,
            comments,
            iteration,
            position_in_batch,
        )

    output_images = []
    infotexts = []
    outputs = []
    comments = {}
    devices.torch_gc()

    with devices.autocast():
        p.init(p.all_prompts, p.all_seeds, p.all_subseeds)

    if state.job_count == -1:
        state.job_count = p.n_iter

    extra_network_data = None
    for n in range(p.n_iter):
        p.iteration = n

    print ("Running Pipeline")

    output = pipeline(
        num_images_per_prompt=10,
        # batch_size=2,
        prompt=p.prompt,
        negative_prompt=p.negative_prompt,
        num_inference_steps=p.steps,
        height=p.height,
        width=p.width,
        # eta=p.eta,
        output_type="pil",
        # styles=p.styles,
    )

    # Save images
    for i, image in enumerate(output.images):
        image.save(image_save_path + f"image_{i+1}.png")

    # image = Image.fromarray(output)
    # images.save_image(image, p.outpath_samples, "", p.seeds[i], p.prompts[i], opts.samples_format, info=infotext(n, i), p=p)

    text = infotext(n, i)
    infotexts.append(text)
    image.info["parameters"] = text
    index_of_first_image = 0
    output_images.append(image)

    result = Processed(
       p,
       images_list=output_images,
       seed=p.all_seeds[0],
       info=infotext(),
       comments="".join(f"{comment}\n" for comment in comments),
       subseed=p.all_subseeds[0],
       index_of_first_image=index_of_first_image,
       infotexts=infotexts,
    )

    return result


class Script(scripts.Script):
    # The title of the script. This is what will be displayed in the dropdown menu.
    def title(self):
        return "Accelerate With Gaudi2"

    # Determines when the script should be shown in the dropdown menu via the
    # returned value. As an example:
    # is_img2img is True if the current tab is img2img, and False if it is txt2img.
    # Thus, return is_img2img to only show the script on the img2img tab.

    def show(self, is_img2img):
        if is_img2img:
            return False
        else:
            return True

    # How the script's is displayed in the UI. See https://gradio.app/docs/#components
    # for the different UI components you can use and how to create them.
    # Most UI components can return a value, such as a boolean for a checkbox.
    # The returned values are passed to the run method as parameters.

    def ui(self, is_img2img):
        # Placeholder for UI Options
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
        processed = process_images_gaudi2(p)

        # processed = process_images(p)
        # use the save_images method from images.py to save
        # for i in range(len(processed.images)):
        #     basename = image_save_path +f "image_{i+1}.png"
        #     images.save_image(image=processed.images[i], path="outputs/txt2img-images", basename=basename, prompt=p.prompt, p=p)
        #     save_image(image, path, basename, seed=None, prompt=None, extension='png', info=None, short_filename=False, no_prompt=False, grid=False, pnginfo_section_name='parameters', p=None, existing_info=None, forced_filename=None, suffix="", save_to_dirs=None):

        return processed
