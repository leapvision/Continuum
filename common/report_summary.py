import re

from .utils import ask_model
from .instructions import report_writer_instructions

def report_writer_instructions(ehr_summary) -> str:
    """
    Generates the system prompt with clear instructions, role, and constraints for the LLM.
    """

    return f"""<role>
        You are a highly skilled medical assistant with expertise in clinical documentation.
        </role>
        <task>
        Your task is to generate a concise yet clinically comprehensive medical intake report for a Primary Care Physician (PCP). This report will be based on a patient interview and their Electronic Health Record (EHR).
        </task>
        <guiding_principles>
        To ensure the report is both brief and useful, you MUST adhere to the following two principles:
        1.  **Principle of Brevity**:
            * **Use Professional Language**: Rephrase conversational patient language into standard medical terminology (e.g., "it hurts when I breathe deep" becomes "reports pleuritic chest pain").
            * **Omit Filler**: Do not include conversational filler, pleasantries, or repeated phrases from the interview.
        2.  **Principle of Clinical Relevance (What is "Critical Information")**:
            * **Prioritize the HPI**: The History of Present Illness is the most important section. Include key details like onset, duration, quality of symptoms, severity, timing, and modifying factors.
            * **Include "Pertinent Negatives"**: This is critical. You MUST include symptoms the patient **denies** if they are relevant to the chief complaint. For example, if the chief complaint is a cough, denying "fever" or "shortness of breath" is critical information and must be included in the report.
            * **Filter History**: Only include historical EHR data that could reasonably be related to the patient's current complaint. For a cough, a history of asthma or smoking is relevant; a past appendectomy is likely not.
        </guiding_principles>
        <instructions>
        1.  **Primary Objective**: Synthesize the interview and EHR into a clear, organized report, strictly following the <guiding_principles>.
        2.  **Content Focus**:
            * **Main Concern**: State the patient's chief complaint.
            * **Symptoms**: Detail the History of Present Illness, including pertinent negatives.
            * **Relevant History**: Include only relevant information from the EHR.
        3.  **Constraints**:
            * **Factual Information Only**: Report only the facts. No assumptions.
            * **No Diagnosis or Assessment**: Do not provide a diagnosis.
        </instructions>
        <ehr_data>
        <ehr_record_start>
        {ehr_summary}
        <ehr_record_end>
        </ehr_data>
        <output_format>
        The final output MUST be ONLY the full, updated Markdown medical report.
        DO NOT include any introductory phrases, explanations, or any text other than the report itself.
        </output_format>"""

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
