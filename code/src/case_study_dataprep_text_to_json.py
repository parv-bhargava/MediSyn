import json
import re
import os
import zipfile
from pathlib import Path


def clean_text(t):
    """Normalize whitespace and clean text content.

        Args:
            t (str): Input text with potential irregular whitespace

        Returns:
            str: Text with single spaces between words and no leading/trailing spaces
    """
    return ' '.join(t.split()).strip()


def extract_patient_info(text):
    """Extract patient information from case study text.

        Args:
            text (str): Full text of the case study

        Returns:
            dict: Dictionary containing name, age, and diagnosis list
    """
    patient_info = {"name": "", "age": "", "diagnosis": []}
    patient_match = re.search(r'Meet The Team\n\n(.+?)\n\n', text, re.DOTALL)
    if patient_match:
        patient_line = patient_match.group(1).strip().split('\n')[0]
        if patient_line:
            name_age = patient_line.split(' ', 1)
            patient_info["name"] = name_age[0].strip()
            patient_info["age"] = name_age[1].strip() if len(name_age) > 1 else ""
            diagnosis_match = re.search(r'Current Diagnosis: (.+?)\n', text)
            if diagnosis_match:
                patient_info["diagnosis"] = [d.strip() for d in diagnosis_match.group(1).split(',')]
    return patient_info


def extract_summary(text):
    """Extract and clean the summary section from case study text.

        Args:
            text (str): Full text of the case study

        Returns:
            str: Cleaned summary text or empty string if not found
    """
    summary_match = re.search(r'Summary\n\n(.+?)\nPatient Info', text, re.DOTALL)
    return clean_text(summary_match.group(1)) if summary_match else ""


def extract_team_members(text):
    """Extract list of team members from case study text.

        Args:
            text (str): Full text of the case study

        Returns:
            list: List of unique team member names, excluding patient info
    """
    team_members = []
    team_section_match = re.search(r'Meet The Team\n\n(.+?)\nFamily', text, re.DOTALL)
    if team_section_match:
        lines = [line.strip() for line in team_section_match.group(1).split('\n') if line.strip()]
        valid_members = [line for line in lines[1:] if not line.startswith("Current Diagnosis:")]
        team_members = list(dict.fromkeys(valid_members))
    return team_members


def extract_background(text):
    """Extract and clean the background section from case study text.

        Args:
            text (str): Full text of the case study

        Returns:
            str: Cleaned background text or empty string if not found
    """
    background_match = re.search(r'Background\n\n(.+?)\nHow They Collaborated', text, re.DOTALL)
    return clean_text(background_match.group(1)) if background_match else ""


def parse_assessment_results(text):
    """Parse assessment results section from case study text.

        Args:
            text (str): Full text of the case study

        Returns:
            dict: Dictionary with general results and role-specific findings
                  Structure: {"general_results": str, "roles": {role: description}}
    """
    section = {"general_results": "", "roles": {}}

    # Find content between Assessment Results and IPP Treatment Plan
    results_match = re.search(
        r'Assessment Results \(Summarize key diagnostic results\)\n\n(.*?)(?=\nIPP Treatment Plan)',
        text,
        re.DOTALL
    )

    if not results_match:
        return section

    content = results_match.group(1)

    # Remove page navigation elements
    cleaned_content = re.sub(r'Go back to Summary|Continue for more|Summary Page \d+ of \d+|Case Rubric \d+ of \d+|Case Rubric continued', '', content)

    # Split into general results and role entries
    parts = re.split(r'\n(?=\w+\s*[\w/]*:)', cleaned_content)

    if parts:
        section["general_results"] = clean_text(parts[0])

        # Process each role entry
        for part in parts[1:]:
            # Split on first colon only
            role_desc = re.match(r'^([^:]+):\s*(.+)$', part, re.DOTALL)
            if role_desc:
                role = role_desc.group(1).strip()
                description = clean_text(role_desc.group(2))
                section["roles"][role] = description

    return section

def parse_assessment_plan(text):
    """Parse assessment plan section from case study text.

        Args:
            text (str): Full text of the case study

        Returns:
            dict: Dictionary with general plan and role-specific responsibilities
                  Structure: {"general_plan": str, "roles": {role: description}}
    """
    section = {"general_plan": "", "roles": {}}

    # Find content between Assessment Plan and Assessment Results
    plan_match = re.search(
        r'Assessment Plan \(Determine roles/ responsibilities for evaluation\)\n\n(.*?)(?=\nAssessment Results)',
        text,
        re.DOTALL
    )

    if not plan_match:
        return section

    content = plan_match.group(1)

    # Remove page navigation elements
    cleaned_content = re.sub(
        r'Go back to Summary|Continue for more|Summary Page \d+ of \d+|Case Rubric \d+ of \d+|Case Rubric continued',
        '',
        content
    )

    # Split into general plan and role entries
    parts = re.split(r'\n(?=\w+\s*[\w/]*:)', cleaned_content)

    if parts:
        section["general_plan"] = clean_text(parts[0])

        # Process each role entry
        for part in parts[1:]:
            # Split on first colon only
            role_desc = re.match(r'^([^:]+):\s*(.+)$', part, re.DOTALL)
            if role_desc:
                role = role_desc.group(1).strip()
                description = clean_text(role_desc.group(2))
                section["roles"][role] = description

    return section

def parse_treatment_plan(text):
    """Parse treatment plan section from case study text.

        Args:
            text (str): Full text of the case study

        Returns:
            dict: Dictionary with general plan and role-specific actions
                  Structure: {"general_plan": str, "roles": {role: description}}
    """
    section = {"general_plan": "", "roles": {}}

    # Find content between IPP Treatment Plan and Treatment Outcomes
    plan_match = re.search(
        r'IPP Treatment Plan \(Discuss, reflect, and modify recommendations to develop a coordinated plan\)\n\n(.*?)(?=Case Rubric continued\n\nTreatment Outcomes)',
        text,
        re.DOTALL
    )

    if not plan_match:
        return section

    content = plan_match.group(1)

    # Remove page navigation elements
    cleaned_content = re.sub(
        r'Go back to Summary|Continue for more|Summary Page \d+ of \d+|Case Rubric \d+ of \d+|Case Rubric continued',
        '',
        content
    )

    # Split into general plan and role entries
    parts = re.split(r'\n(?=\w+\s*[\w/]*:)', cleaned_content)

    if parts:
        section["general_plan"] = clean_text(parts[0])

        # Process each role entry
        for part in parts[1:]:
            # Split on first colon only and stop at newline
            role_desc = re.match(r'^([^:]+):\s*([^\n]*)', part)
            if role_desc:
                role = role_desc.group(1).strip()
                description = clean_text(role_desc.group(2))
                section["roles"][role] = description

    return section

def parse_case_study(text):
    """Main function to parse complete case study from text.

        Args:
            text (str): Full text content of the case study

        Returns:
            dict: Structured case study data with keys:
                  - patient_info
                  - summary
                  - team_members
                  - background
                  - assessment_plan
                  - assessment_results
                  - treatment_plan
    """
    return {
        "patient_info": extract_patient_info(text),
        "summary": extract_summary(text),
        "team_members": extract_team_members(text),
        "background": extract_background(text),
        "assessment_plan": parse_assessment_plan(text),
        "assessment_results": parse_assessment_results(text),
        "treatment_plan": parse_treatment_plan(text)  # Use new function
    }


def process_zip(zip_path, output_dir="Case_Study_json"):
    """Process zip archive containing multiple case study text files.

        Args:
            zip_path (str): Path to input zip file
            output_dir (str): Directory to save JSON outputs (default: "Case_Study_json")

        Raises:
            FileNotFoundError: If input zip file doesn't exist
            zipfile.BadZipFile: If input file is not a valid zip archive
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            if file_info.filename.lower().endswith('.txt'):
                # Read text file content from zip
                with zip_ref.open(file_info) as file:
                    text_content = file.read().decode('utf-8')

                    # Process with existing parser
                    json_data = parse_case_study(text_content)

                    # Create output path
                    json_filename = Path(file_info.filename).stem + '.json'
                    output_path = Path(output_dir) / json_filename

                    # Save JSON output
                    with open(output_path, 'w') as json_file:
                        json.dump(json_data, json_file, indent=2)

# Usage remains the same
print(os.getcwd())
os.chdir("../../../Case_Study_Data")

input_zip = "case-study-text.zip"  # Replace with your zip file path
process_zip(input_zip)
print(f"Processed all files in {input_zip} and saved JSONs to Case_Study_json directory")