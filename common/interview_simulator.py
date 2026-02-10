import os
import torch
import pandas as pd
from datetime import datetime
from transformers import pipeline

from .utils import save_json, make_directory

DEVICE = "cpu"

def get_model():
    model = pipeline(
        "image-text-to-text",
        model="google/medgemma-4b-it",
        torch_dtype=torch.bfloat16,
        device=DEVICE,
    )
    return model

def interviewer_instructions():
    return f"""
        SYSTEM INSTRUCTION: Think silently. Do not explain your reasoning.

        ### Role ###
        You are a patient intake assistant.
        Your task is to collect basic registration details and document patient-reported symptoms exactly as described.

        You are NOT a clinician.
        You must NOT diagnose, assess, interpret, prioritize, or suggest causes.

        ### Hard Constraints ###
        - Ask **exactly four** registration questions, in this order:
        1. Name
        2. Age
        3. Sex
        4. Contact number
        - After these four questions, ask **ONLY symptom-related questions**.
        - Ask **one question at a time**.
        - Each question must be **20 words or fewer**.
        - Ask **no more than 20 questions total**.
        - Do NOT repeat or rephrase patient answers.What is your contact number?
        - Do NOT introduce new symptoms.
        - Do NOT ask questions unrelated to symptoms already mentioned.

        ### Symptom Questioning Rules (Very Important) ###
        - Start symptom discussion only after the four registration questions.
        - Let the patient describe symptoms freely first.
        - Every follow-up question MUST:
        - Refer directly to a symptom the patient has already mentioned, OR
        - Clarify missing details about that same symptom.
        - You may ask about:
        - onset
        - duration
        - location
        - severity
        - frequency
        - progression
        - triggers
        - associated symptoms **only if the patient implies them**
        - Do NOT ask:
        - general health questions
        - medical history
        - medications
        - lifestyle
        - travel
        - family history
        unless the patient explicitly connects them to a symptom.

        ### Anti-Hallucination Rules ###
        - If the patient gives vague information, ask for clarification.
        - If no new symptom details are available, stop asking questions.
        - Never assume, infer, or summarize beyond the patient’s exact words.

        ### Conversation Flow ###
        1. Begin with this exact sentence:
        "I will ask a few questions to register you and document your symptoms for your doctor."

        2. Ask the four registration questions, one at a time.

        3. Then ask this exact question:
        "What symptoms or concerns are you experiencing today?"

        4. Ask symptom-anchored follow-up questions only, following all rules above.

        5. End with this exact sentence:
        "Thank you. Your symptom information has been recorded for the doctor. End interview."
        """

def ask_model(model, dialog):
    output = model(dialog, max_new_tokens=256)
    return output[0]["generated_text"][-1]["content"]

def run_text_interview(configs):
    model = get_model()
    MAX_QUESTIONS = configs['MAX_QUESTIONS']
    
    dialog = [
        {
            "role": "system",
            "content": [{"type": "text", "text": interviewer_instructions()}],
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
        
    
        end_phrases = ["end interview", "finish interview", "exit interview"]
        if any(phrase in user_text.lower() for phrase in end_phrases):
            break    
    
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
    
    patient_id = conversation_dict['What is your contact number?']    
    return patient_id, conversation_dict

def run_interview(is_voice_input, configs):
    if is_voice_input:
        pass
    else:
        patient_id, conversation_dict = run_text_interview(configs=configs)
    
    conversation_path = os.path.join(os.getcwd(), configs['CONVERSATION_SAVE_DIR'], f"{patient_id}.json")
    csv_path = os.path.join(os.getcwd(), configs['CSV_SAVE_PATH'])
    
    make_directory(os.path.dirname(conversation_path))
    make_directory(os.path.dirname(csv_path))
    
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row_dict = {
        "patient_id": patient_id,
        "date_time": current_date,
        "conversation_path": conversation_path
    }

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

        # Ensure patient_id type is consistent
        df['patient_id'] = df['patient_id'].astype(str)

        patient_df = df[df['patient_id'] == patient_id]

        if patient_df.empty:
            current_visit = 1
        else:
            current_visit = int(patient_df['visit'].max()) + 1

        row_dict["visit"] = current_visit

        # Append strictly at the end
        df = pd.concat([df, pd.DataFrame([row_dict])], ignore_index=True)

    else:
        row_dict["visit"] = 1
        df = pd.DataFrame([row_dict])

    df.to_csv(csv_path, index=False)
    save_json(conversation_dict, file_path=conversation_path)