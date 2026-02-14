import os
import torch
import pandas as pd
from datetime import datetime
from transformers import pipeline

from common.report_summary import write_report
from common.instructions import intake_system_instructions
from common.text_interview_simulator import run_intake_text_interview
from common.utils import get_input_mode, read_json, get_patient_details, make_directory, get_ehr_summary, save_json, save_txt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
    
    dataset_schema = {
        "patient_id" : None,
        "date_time" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "visit" : None,
        "conversation_path" : None,
        "report_path": None
    }
    
    conversation_schema = {
        "patient_detail": None,
        "symtoms": None
    }
    
    configs = read_json(os.path.join(os.path.join(os.getcwd(), "config.json")))
    configs = {key : os.path.join(os.getcwd(), value) for key, value in configs.items()}
    
    csv_path = configs['CSV_SAVE_PATH']
    conversation_dir = configs['CONVERSATION_SAVE_DIR']
    report_dir = configs['reports_dir']
    make_directory(os.path.join(os.getcwd(), conversation_dir))
    make_directory(os.path.join(os.getcwd(), os.path.dirname(csv_path)))
    make_directory(os.path.join(os.getcwd(), report_dir))
        
    # load_model()
    is_voice_input = get_input_mode()
    if is_voice_input:
        pass
    else:
        patient_details, last_visit_id, existing_report, ehr_records, patient_records_df = get_patient_details(dataset_schema=dataset_schema, csv_path=csv_path)
        
        ehr_summary = get_ehr_summary(patient_name = patient_details['name'], ehr_records=ehr_records, model=MODEL)

        intake_instructions = intake_system_instructions(patient_name=patient_details['name'], ehr_summary=ehr_summary)
        symtoms_conversation_dict = run_intake_text_interview(instructions=intake_instructions, model=MODEL)
        
        report_summary = write_report(model=MODEL, ehr_summary=ehr_summary, interview_text=symtoms_conversation_dict)
        
        patient_id = patient_details['patient_id']
        current_visit = last_visit_id+1
        conversation_path = os.path.join(conversation_dir, f"{patient_id}_{current_visit}.json")
        report_path = os.path.join(report_dir, f"{patient_id}_{current_visit}.txt")
        
        dataset_schema["patient_id"] = patient_id
        dataset_schema["visit"] = current_visit
        dataset_schema["conversation_path"] = conversation_path
        dataset_schema["report_path"] = report_path
        conversation_schema['patient_detail'] = patient_details
        conversation_schema["symtoms"] = symtoms_conversation_dict
        
        save_json(data=conversation_schema, file_path=conversation_path)
        save_txt(data=report_summary, file_path=report_path)
        
        new_row_df = pd.DataFrame([dataset_schema])
        if patient_records_df.empty:
            patient_records_df = new_row_df
        else:
            patient_records_df = pd.concat([patient_records_df, new_row_df])

        patient_records_df.to_csv(csv_path, index=False)
        
        

    
    # patient_id  = "9328814419"
     
    # get_report_summary(configs=configs, patient_id=patient_id, current_conversation_dict = "conversation_dict")
    
    