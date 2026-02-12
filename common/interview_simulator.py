import os
import torch
import warnings
import pandas as pd
from datetime import datetime
warnings.filterwarnings("ignore")

from .instructions import interviewer_instructions
from .utils import save_json, make_directory, ask_model

def get_personal_detail():
    patient_details = {
        "name": None,
        "age": None,
        "sex": None,
        "patient_id": None
    }
    
    print("\nI will ask a few questions to register you and document your symptoms for your doctor.")
    patient_details["patient_id"] = input("\nWhat is your contact number? \n")
    patient_details["name"] = input("\nWhat is your name? \n")
    patient_details["age"] = input("\nHow old are you? \n")
    patient_details["sex"] = input("\nWhat is your sex? \n")
    return patient_details
    
def run_text_interview(instructions, model):
    MAX_QUESTIONS = 20
    
    dialog = [
        {
            "role": "system",
            "content": [{"type": "text", "text": instructions}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "start interview"}],
        },
    ]

    question_count = 0

    conversation_dict = {}
    while question_count < MAX_QUESTIONS:
        assistant_text = ask_model(model, dialog)
        print(f"\nAssistant: {assistant_text}")

        # Append assistant message
        dialog.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
            }
        )

        question_count += 1

        # Model decided to end early
        if "End interview." in assistant_text:
            break

        # Patient response
        user_text = input("\nPatient: ")
       
        conversation_dict[assistant_text.strip("\n")] = user_text.strip("\n")

        dialog.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": user_text}],
            }
        )

    # Hard stop if model failed to end properly
    if question_count >= MAX_QUESTIONS:
        print(
            "\nAssistant: Thank you for answering my questions. "
            "I have everything needed to prepare a report for your visit. End interview."
        )
    
    return conversation_dict

def run_interview(is_voice_input, configs, model):
    if is_voice_input:
        pass
    else:
        patient_details = get_personal_detail()
        symtoms_conversation_dict = run_text_interview(instructions=interviewer_instructions(), model=model)
    
    conversation_dict = {}
    conversation_dict['patient_detail'] = patient_details
    conversation_dict['symtoms'] = symtoms_conversation_dict
    
    patient_id = patient_details['patient_id']
    csv_path = configs['CSV_SAVE_PATH']
    conversation_dir = configs['CONVERSATION_SAVE_DIR']
    
    make_directory(os.path.dirname(conversation_dir))
    make_directory(os.path.dirname(csv_path))
    
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row_dict = {
        "patient_id": patient_id,
        "date_time": current_date,
    }

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, dtype={"patient_id": str})
        
        patient_df = df[df['patient_id'] == patient_id]

        current_visit = int(patient_df['visit'].max()) + 1 if not patient_df.empty else 1
    else:
        df = pd.DataFrame()
        current_visit = 1

    row_dict["visit"] = current_visit
    conversation_path = os.path.join(conversation_dir, f"{patient_id}_{current_visit}.json")
    row_dict['conversation_path'] = conversation_path

    df = pd.concat([df, pd.DataFrame([row_dict])], ignore_index=True)

    save_json(conversation_dict, file_path=conversation_path)    
    df.to_csv(csv_path, index=False)
    return patient_id, conversation_dict