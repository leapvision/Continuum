import os
import json
import pandas as pd
from dotenv import load_dotenv

load_dotenv()  # loads .env once at import time

def get_input_mode():
    """
    Ask the user whether they want to proceed with Voice input or Text input.

    Returns:
        bool: True if user chooses Voice, False if user chooses Text.
    """
    while True:
        # Prompt user for input
        choice = input("Do you want to proceed with Text or Voice? (T/V): ").strip().lower()

        # User chooses Text → return False
        if choice in ['t', 'text']:
            return False

        # User chooses Voice → return True
        elif choice in ['v', 'voice']:
            return True

        # Invalid input → ask again
        else:
            print("Invalid input. Please type 'T' for Text or 'V' for Voice.")

def get_env_var(key: str, default=None, required: bool = False):
    """
    Fetch an environment variable from .env or system env.

    Args:
        key (str): Environment variable name
        default: Default value if not found
        required (bool): Raise error if variable is missing

    Returns:
        str | None
    """
    value = os.getenv(key, default)

    if required and value is None:
        raise RuntimeError(f"Missing required environment variable: {key}")

    return value

def make_directory(dir_path: str) -> None:
    """
    Create a directory if it does not already exist.

    Args:
        dir_path (str): Path of the directory to create.

    Notes:
        - If the directory already exists, this function does nothing.
        - Creates intermediate directories as needed.
    """
    # Create the directory and all intermediate directories if needed
    os.makedirs(dir_path, exist_ok=True)
    
def read_json(file_path: str) -> dict:
    """
    Reads a JSON file and returns its contents as a Python dictionary.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict: Contents of the JSON file, or empty dict on error.
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: File '{file_path}' contains invalid JSON.")
        return {}

def save_json(data: dict, file_path: str) -> None:
    """
    Saves a Python dictionary to a JSON file.

    Args:
        data (dict): The dictionary to save.
        file_path (str): Path to the JSON file to write.
    """
    try:
        with open(file_path, "w") as f:
            # Pretty-print JSON with 4-space indentation
            json.dump(data, f, indent=6)
        print(f"Data successfully saved to '{file_path}'")
    except Exception as e:
        print(f"Error saving JSON to '{file_path}': {e}")

def read_txt(file_path):
    with open(file_path, "r") as f:
        data = f.read()
    return data

def save_txt(data, file_path):
    with open(file_path, "w") as f:
        f.write(data)

def ask_model(model, dialog):
    """
    Sends a dialog to the language model and returns the final generated response.

    Assumes:
    - `model(dialog)` returns a list of results
    - The first result contains a "generated_text" field
    - "generated_text" is a list of messages
    - The model's final reply is the last element's "content"

    Parameters:
        model (callable): The language model inference function
        dialog (list | dict): Structured conversation input for the model

    Returns:
        str: The content of the model's final generated message
    """
    output = model(dialog)
    return output[0]["generated_text"][-1]["content"]

def collect_new_patient_details(patient_id):
    return {
        "patient_id": patient_id,
        "name": input("\nWhat is your name?\n").strip(),
        "age": int(input("\nHow old are you?\n")),
        "sex": input("\nWhat is your sex?\n").strip(),
    }

def get_patient_details(dataset_schema, csv_path):

    patient_id = input("\nPlease enter your Contact number :\n").strip()
    if not patient_id:
        raise ValueError("Patient ID cannot be empty.")

    patient_details = {}
    last_visit = 0
    exising_report = None
    df = pd.DataFrame()

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

        required_cols = {"patient_id", "visit", "conversation_path"}
        if not required_cols.issubset(df.columns):
            raise ValueError("CSV schema is invalid or corrupted.")

        df["patient_id"] = df["patient_id"].astype(str)
        patient_rows = df[df["patient_id"] == patient_id]

        if not patient_rows.empty:
            print("\nWe found your record. Retrieving details from your last visit...\n")

            patient_rows["visit"] = pd.to_numeric(patient_rows["visit"], errors="coerce")
            ehr_records = [(visit, read_txt(report_path))for visit, report_path in zip(patient_rows['visit'].tolist(), patient_rows['report_path'].tolist())]
            
            last_visit = int(patient_rows["visit"].max())

            last_visit_row = patient_rows.loc[patient_rows["visit"].idxmax()]

            json_path = last_visit_row["conversation_path"]

            if pd.isna(json_path) or not os.path.exists(json_path):
                raise FileNotFoundError(f"Conversation file missing for patient {patient_id}")

            record = read_json(json_path)
            patient_details = record.get("patient_detail", {})

        else:
            print("\nNo existing record found. Let's collect your details.\n")
            patient_details = collect_new_patient_details(patient_id)
            ehr_records = "There is No Past Records"
    else:
        print("\nNo records found. Let's collect your details.\n")
        patient_details = collect_new_patient_details(patient_id)
        ehr_records = "There is No Past Records"

    dataset_schema["patient_id"] = patient_id
    dataset_schema["visit"] = last_visit + 1

    return patient_details, last_visit, exising_report, ehr_records, df

def get_ehr_summary(patient_name, ehr_records, model):
    
    if ehr_records == "There is No Past Records":
        ehr_summary = ehr_records
    else:
        ehr_summary = ask_model(dialog=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": f"""You are a medical assistant summarizing the EHR (FHIR) records for the patient {patient_name}.
                        Provide a concise summary of the patient's medical history, including any existing conditions, medications, and relevant past treatments.
                        Do not include personal opinions or assumptions, only factual information. Data is in list of tuple format. where first element is visit id and second is report."""
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": ehr_records
                    }
                ]
            }
        ], model=model)
    return ehr_summary