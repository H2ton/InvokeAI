import base64
from json import dumps
from io import BytesIO

from ldm.generate import Generate
from ldm.invoke.pngwriter import PngWriter
from flask import Flask, request

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import base64
from PIL import Image
import io
import matplotlib.pyplot as plt
from torch import autocast 
import numpy as np

DEVICE = "cuda"

app = FastAPI()
t2i = None
image_quality = 84

image_list = list()

def init():
    global t2i
    
    t2i = Generate()

def im_2_b64(image):
    buff = io.BytesIO()
    image.save(buff, format="jpeg", optimize=True, quality=image_quality)
    img_str = base64.b64encode(buff.getvalue())
    return img_str

def b64_2_img(data):
    buff = io.BytesIO(base64.b64decode(data))
    return Image.open(buff)

class Txt2ImgIn(BaseModel):
    seed: int = 100
    width: int = 512
    height: int = 512
    prompt: str = None
    step: int = 100
    scale: float = 7.5
    eta: float = 0.0
    sampler_name: str = 'k_euler'
    iterations = 1
    img_quality = 84

class Txt2ImgOut(BaseModel):
    image_num: int = 1
    images_base64: list = None

@app.post("/txt2img/")
async def Inpainting(inpaint: Txt2ImgIn):

    global image_quality
    image_quality = inpaint.img_quality

    image_list.clear()

    results = t2i.prompt2image(
        prompt=inpaint.prompt,
        outdir="./outputs",
        image_callback=None,
        width = inpaint.width,
        height = inpaint.height,
        seed = inpaint.seed,
        sampler_name = inpaint.sampler_name,
        steps = inpaint.step,
        cfg_scale = inpaint.scale,
        ddim_eta = inpaint.eta,
        iterations = inpaint.iterations
    )

    out = Txt2ImgOut()
    out.images_base64 = list()

    for i in range(len(results)):
        out.images_base64.append(im_2_b64(results[i][0])) # [n][0] is image, [n][1] is seed

    out.image_num = len(results)

    return out

class InpaintIn(BaseModel):
    base_base64: str
    mask_base64: str
    seed: int = 100
    width: int = 512
    height: int = 512
    prompt: str = None
    step: int = 100
    scale: float = 7.5
    strength: float = 0.8
    eta: float = 0.0
    sampler_name: str = 'ddim'
    fit = False
    iterations = 1
    img_quality = 84

class InpaintOut(BaseModel):
    image_num: int = 1
    images_base64: list = None

def image_writer(image, seed, first_seed=0):
    image_list.append(im_2_b64(image))

@app.post("/img2img/")
async def Inpainting(inpaint: InpaintIn):

    global image_quality
    image_quality = inpaint.img_quality

    init_img = b64_2_img(inpaint.base_base64)
    init_img = init_img.resize((inpaint.width, inpaint.height))

    mask_img = b64_2_img(inpaint.mask_base64)
    mask_img = mask_img.resize((inpaint.width, inpaint.height))

    mask_img = mask_img.convert('RGBA')
    px = mask_img.load()

    for i in range(0, mask_img.width):
        for j in range(0, mask_img.height):
            px[i, j] = (px[i, j][0],px[i, j][1],px[i, j][2],255 - px[i, j][2])

    image_list.clear()

    results = t2i.prompt2image(
        prompt=inpaint.prompt,
        outdir="./outputs",
        image_callback=None,
        width = inpaint.width,
        height = inpaint.height,
        seed = inpaint.seed,
        sampler_name = inpaint.sampler_name,
        steps = inpaint.step,
        cfg_scale = inpaint.scale,
        init_img = init_img,
        init_mask = mask_img,
        fit = inpaint.fit,
        strength = inpaint.strength,
        init_color = None,
        ddim_eta = inpaint.eta,
        iterations = inpaint.iterations
    )

    out = InpaintOut()
    out.images_base64 = list()

    for i in range(len(results)):
        out.images_base64.append(im_2_b64(results[i][0])) # [n][0] is image, [n][1] is seed

    out.image_num = len(results)

    return out

if __name__ == "__main__":
    init()
    uvicorn.run(app, host="0.0.0.0", port=8000)