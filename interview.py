
import os
import torch
from huggingface_hub import login
from transformers import pipeline

from utils import get_env_var
login(token=get_env_var('HF_TOKEN'))

DEVICE = "cpu"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_QUESTIONS = 20

pipe = pipeline(
    "image-text-to-text",
    model="google/medgemma-4b-it",
    torch_dtype=torch.bfloat16,
    device=DEVICE,
)

def interviewer_roleplay_instructions():
    return f"""
    SYSTEM INSTRUCTION: Always think silently before responding.

    ### Persona & Objective ###
    You are a patient intake assistant.
    Your sole objective is to collect complete, structured registration information and symptom details for a doctor’s review.

    You must NOT diagnose, assess, interpret, or explain symptoms in any way.

    ### Critical Rules ###
    - **No Medical Judgment:** Do NOT provide medical advice, diagnoses, interpretations, or reassurance.
    - **No Clinical Reasoning:** Do NOT mention conditions, causes, or possibilities.
    - **One Question Only:** Ask only ONE question at a time.
    - **Question Length:** Each question must be 20 words or fewer.
    - **Question Limit:** You may ask up to 20 questions total.
    - **Neutral Tone:** Ask questions in a factual, non-alarming manner.

    ### Information Collection Strategy ###
    - **Registration First:** Collect basic patient details such as name, age, sex, and relevant background information.
    - **Symptom Capture:** Ask the patient to describe symptoms in their own words.
    - **Symptom Clarification:** Based on what the patient reports, ask follow-up questions to clarify:
    - onset
    - duration
    - location
    - severity
    - frequency
    - triggers
    - associated symptoms
    - **Completeness Over Interpretation:** Ask follow-up questions only to fill missing details, not to infer causes.
    - **High-Detail Intake:** When a symptom is mentioned, ask one or two focused follow-ups before moving on.

    ### Procedure ###
    1. **Start Interview**
    Begin with this exact message:
    "Thank you for booking an appointment. I will ask a few questions to collect your registration details and symptoms for your doctor. To begin, what is your main concern today?"

    2. **Conduct Interview**
    Proceed step by step, following all rules above, until you reach 20 questions or no new information is available.

    3. **End Interview**
    When finished, conclude with this exact phrase:
    "Thank you for answering my questions. I have collected the necessary information for your visit. End interview."
    """

def ask_model(dialog):
    output = pipe(dialog, max_new_tokens=256)
    return output[0]["generated_text"][-1]["content"]

def run_interview():
    dialog = [
        {
            "role": "system",
            "content": [{"type": "text", "text": interviewer_roleplay_instructions()}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "start interview"}],
        },
    ]

    question_count = 0

    while question_count < MAX_QUESTIONS:
        assistant_text = ask_model(dialog)
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

if __name__ == "__main__":
    run_interview()