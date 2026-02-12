import os
import json
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