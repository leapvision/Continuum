import re
from .utils import ask_model
from .instructions import report_writer_instructions

def write_report(model, ehr_summary, interview_text) -> str:
    """
    Constructs the full prompt, sends it to the LLM, and processes the response.
    This function handles both the initial creation and subsequent updates of a report.
    """
    # Generate the detailed system instructions
    instructions = report_writer_instructions(ehr_summary=ehr_summary)

    # Construct the user prompt with the specific task and data
    user_prompt = f"""<interview_start>
    {interview_text}
    <interview_end>
    <previous_report>
    {ehr_summary}
    </previous_report>
    <task_instructions>
    Update the report in the `<previous_report>` tags using the new information from the `<interview_start>` section.
    1.  **Integrate New Information**: Add new symptoms or details from the interview into the appropriate sections.
    2.  **Update Existing Information**: If the interview provides more current information, replace outdated details.
    3.  **Maintain Conciseness**: Remove any information that is no longer relevant.
    4.  **Preserve Critical Data**: Do not remove essential historical data (like Hypertension) that could be vital for diagnosis, but ensure it is presented concisely under "Relevant Medical History".
    5.  **Adhere to Section Titles**: Do not change the existing Markdown section titles.
    </task_instructions>
    Now, generate the complete and updated medical report based on all system and user instructions. Your response should be the Markdown text of the report only."""

    # Assemble the full message payload for the LLM API
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": instructions}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": user_prompt}]
        }
    ]

    report = ask_model(model=model, dialog=messages)
    cleaned_report = report.strip()

    # The LLM sometimes wraps the markdown report in a markdown code block.
    # This regex checks if the entire string is a code block and extracts the content.
    match = re.match(r'^\s*```(?:markdown)?\s*(.*?)\s*```\s*$', cleaned_report, re.DOTALL | re.IGNORECASE)
    if match:
        cleaned_report = match.group(1)

    return cleaned_report.strip()
