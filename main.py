import os
import torch
from transformers import pipeline

from common.utils import get_input_mode, read_json
from common.interview_simulator import run_interview
from common.report_summary import get_report_summary

DEVICE = 'cuda'

MODEL = None
def load_model():
    global MODEL
    MODEL = pipeline(
        "image-text-to-text",
        model="google/medgemma-4b-it",
        torch_dtype=torch.bfloat16,
        device=DEVICE,
    )

if __name__ == "__main__":
    
    load_model()
    is_voice_input = get_input_mode()
    
    configs = read_json(os.path.join(os.path.join(os.getcwd(), "config.json")))
    configs = {key : os.path.join(os.getcwd(), value) for key, value in configs.items()}

    patient_id, conversation_dict = run_interview(is_voice_input, configs=configs, model=MODEL)
    
    # get_report_summary(configs=configs, patient_id=patient_id, current_conversation_dict = conversation_dict)
    
    