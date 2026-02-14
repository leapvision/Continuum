import warnings
warnings.filterwarnings("ignore")
from .utils import ask_model

def run_intake_text_interview(instructions, model):
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
