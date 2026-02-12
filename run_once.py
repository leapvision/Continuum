import torch
from transformers import pipeline
from huggingface_hub import login

from common.utils import get_env_var
login(token=get_env_var('HF_TOKEN'))

MODEL = pipeline(
    "image-text-to-text",
    model="google/medgemma-4b-it",
    torch_dtype=torch.bfloat16,
)

## Run this code once. 
## After Runnig this code it will download model, and save into .cache folder in ubuntu

## export HF_HUB_OFFLINE=1  ## for offline inference
## export HF_HOME= <.cache Path>/huggingface



