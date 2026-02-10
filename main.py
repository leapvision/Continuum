import os
from huggingface_hub import login

from common.utils import get_env_var, get_input_mode, read_json
from common.interview_simulator import run_interview

if __name__ == "__main__":
    
    login(token=get_env_var('HF_TOKEN'))
    
    is_voice_input = get_input_mode()
    
    configs = read_json(os.path.join(os.path.join(os.getcwd(), "config.json")))
    
    run_interview(is_voice_input, configs=configs)