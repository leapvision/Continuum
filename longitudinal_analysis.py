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


def get_patient_images_from_folder(root_folder):
    patient_dir = Path(root_folder).expanduser()
    if not patient_dir.exists():
        raise FileNotFoundError(f"Root folder not found: {patient_dir}")
    if not patient_dir.is_dir():
        raise NotADirectoryError(f"Root path is not a directory: {patient_dir}")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    patient_images = {}

    image_paths = sorted(
        [
            p
            for p in patient_dir.iterdir()
            if p.is_file() and p.suffix.lower() in image_extensions
        ]
    )
    if image_paths:
        print(f"Patient folder: {patient_dir}")
        for image_path in image_paths:
            print(f"  image: {image_path}")
        patient_images[patient_dir.name] = [Image.open(path) for path in image_paths]
        print(f"  total images: {len(image_paths)}")

    if not patient_images:
        raise RuntimeError(f"No patient folders with images found in: {patient_dir}")

    return patient_images


prompt_template = """Compare the provided {num_images} images from the same patient and describe any potential progression of a disease or finding.
"""


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


preprocess_image = True  # @param {type: "boolean"}

patients_root = (
    "~/Drongo/data/test data/longitudinal analysis/mimic_dataset_longitudinal/50022785/"
)
patient_images = get_patient_images_from_folder(patients_root)

for patient_id, images in patient_images.items():
    if preprocess_image:
        processed_images = []
        for image in images:
            # Convert each input image to a square numpy array and normalize pixel values.
            image_array = (pad_image_to_square(image) * 255).astype(np.uint8)
            # Convert the numpy array back to a PIL Image.
            processed_images.append(Image.fromarray(image_array))
        images = processed_images

    prompt = prompt_template.format(num_images=len(images))
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    output = pipe(text=messages, max_new_tokens=2000)
    print(f"\nPatient: {patient_id}")
    print(output[0]["generated_text"][-1]["content"])
