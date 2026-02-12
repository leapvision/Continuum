import os
from pathlib import Path
from transformers import pipeline
from PIL import Image
import torch
import numpy as np
import skimage
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise RuntimeError("HF_TOKEN is missing. Add it to your .env file.")

pipe = pipeline(
    "image-text-to-text",
    model="google/medgemma-1.5-4b-it",
    torch_dtype=torch.bfloat16,
    device="mps",
    token=hf_token,
)

pipe.model.generation_config.do_sample = False


def load_image(path_str):
    image_path = Path(path_str).expanduser()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return Image.open(image_path)


image1 = load_image(
    "~/Drongo/data/test data/longitudinal analysis/3b8b1b7d-054490d5-385641e7-ff43d2c8-9505f058.jpg"
)
image2 = load_image(
    "~/Drongo/data/test data/longitudinal analysis/ed9c0dfc-ea25b576-0f8cc069-df4cdf14-0cd60eb7.jpg"
)


prompt = f"""Provide a comparison of these two images from the same patient and include details on progression of a disease/finding"""


def pad_image_to_square(image_array):
    # Convert image to unsigned byte format and handle grayscale/RGBA images.
    image_array = skimage.util.img_as_ubyte(image_array)
    if len(image_array.shape) < 3:
        image_array = skimage.color.gray2rgb(image_array)
    if image_array.shape[2] == 4:
        image_array = skimage.color.rgba2rgb(image_array)

    # Pad the image to a square shape.
    h = image_array.shape[0]
    w = image_array.shape[1]
    max_dim = max(h, w)
    if h < w:
        dh = w - h
        image_array = np.pad(image_array, ((dh // 2, dh - dh // 2), (0, 0), (0, 0)))
    if w < h:
        dw = h - w
        image_array = np.pad(image_array, ((0, 0), (dw // 2, dw - dw // 2), (0, 0)))
    return image_array


preprocess_image = False  # @param {type: "boolean"}

if preprocess_image:
    # Convert the input image to a square numpy array and normalize pixel values.
    image_array1 = (pad_image_to_square(image1) * 255).astype(np.uint8)
    # Convert the numpy array back to a PIL Image.
    image1 = Image.fromarray(image_array1)
    # Convert the input image to a square numpy array and normalize pixel values.
    image_array2 = (pad_image_to_square(image2) * 255).astype(np.uint8)
    # Convert the numpy array back to a PIL Image.
    image2 = Image.fromarray(image_array2)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image1},
            {"type": "image", "image": image2},
            {"type": "text", "text": prompt},
        ],
    }
]

output = pipe(text=messages, max_new_tokens=2000)
print(output[0]["generated_text"][-1]["content"])
