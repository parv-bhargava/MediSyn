import os
import json
import re
import zipfile
from pathlib import Path


def clean_text(t):
    """Normalize whitespace and clean text content."""
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
            patient_info["name"] = "" if name_age[0][0].isdigit() else name_age[0].strip()
            if len(name_age) > 2:
                patient_info["age"] = name_age[1].strip()
            else:
                age_match = re.search(r'(\d+)[- ]?(YEAR|MONTH|WEEKS)', text)
                if age_match:
                    patient_info["age"] = f"{age_match.group(1)} {age_match.group(2).lower()} old"
            diagnosis_match = re.search(r'Current Diagnosis: (.+?)\n', text)
            if diagnosis_match:
                patient_info["diagnosis"] = [d.strip() for d in diagnosis_match.group(1).split(',')]
    return patient_info


def extract_summary(text):
    """Extract and clean the summary section."""
    summary_match = re.search(r'Summary\s+(.+?)\s+Patient Info', text, re.DOTALL)
    return clean_text(summary_match.group(1)) if summary_match else ""


def extract_team_members(text):
    """Extract list of team members from case study text.

        Args:
            text (str): Full text of the case study

        Returns:
            list: List of unique team member names, excluding patient info
    """
    end_markers = (
        "Continue for more",
        "Summary Page",
        "Case Rubric",
        "Patient Info",
        "Background",
        "How They Collaborated",
        "Go to Case Rubric",
        "Acknowledgement",
        "Citations"
    )

    # Build a regex that:
    # - Finds "Meet The Team" (ignoring case) followed by any whitespace/newlines.
    # - Then captures non-greedily everything until a newline followed by one of the
    #   end markers or the end-of-string.
    pattern = (
        r"Meet\s+The\s+Team\s*(?:\n|$)"  # starting marker (with flexible whitespace/newlines)
        r"(.*?)"  # capture group (non-greedy) for team members block
        r"(?=\n(?:{})(?:\b|:)|$)".format("|".join(map(re.escape, end_markers)))
    )

    # Use re.IGNORECASE and re.DOTALL so that the block can span multiple lines.
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if not match:
        return []  # no team section found

    team_block = match.group(1)

    # Split the captured block into lines and filter out empty lines.
    # (Sometimes extra navigation or whitespace lines may appear.)
    lines = [line.strip() for line in team_block.splitlines() if line.strip()]

    # Remove duplicates while preserving order.
    seen = set()
    team_members = []
    for member in lines:
        if member not in seen:
            seen.add(member)
            team_members.append(member)
    age_pattern = re.compile(
        r'(?i)\b\d+\s*-?\s*(?:YEAR(?:S)?-?\s*OLD|MONTH(?:S)?\s*OLD|WEEK(?:S)?)\b',re.IGNORECASE)
    filtered_members = []
    for member in team_members:
        # Skip if the line contains an age indicator (e.g., "4-YEAR OLD")
        if age_pattern.search(member):
            continue
        # Skip if the line mentions "Current Diagnosis"
        if "current diagnosis" in member.lower():
            continue
        filtered_members.append(member)

    return filtered_members


def extract_background(text):
    """Extract background section with improved boundary detection."""
    background_match = re.search(
        r'Background\s+(.+?)(?=\s+How They Collaborated|Assessment Plan)',
        text,
        re.DOTALL
    )
    return clean_text(background_match.group(1)) if background_match else ""


def parse_assessment_plan(text):
    """Parse assessment plan section with flexible boundaries."""
    plan_match = re.search(
        r'Assessment Plan \(Determine roles/ responsibilities for evaluation\)\s+(.+?)(?=\s+Assessment Results|Treatment Outcomes)',
        text,
        re.DOTALL
    )
    if plan_match:
        content = plan_match.group(1)
        cleaned = re.sub(r'Go back to Summary|Continue for more|Case Rubric \d+ of \d+', '', content)
        return clean_text(cleaned)
    return ""


def parse_assessment_results(text):
    """Parse assessment results section with improved pattern."""
    results_match = re.search(
        r'Assessment Results \(Summarize key diagnostic results\)\s+(.+?)(?=\s+IPP Treatment Plan|Treatment Outcomes)',
        text,
        re.DOTALL
    )
    if results_match:
        content = results_match.group(1)
        cleaned = re.sub(r'Case Rubric continued|Continue for more', '', content)
        return clean_text(cleaned)
    return ""


def parse_treatment_plan(text):
    """Parse treatment plan section with updated boundaries."""
    plan_match = re.search(
        r'IPP Treatment Plan \(Discuss, reflect, and modify recommendations to develop a coordinated plan\)\s+(.+?)(?=\s+Treatment Outcomes|Team Follow-Up)',
        text,
        re.DOTALL
    )
    if plan_match:
        content = plan_match.group(1)
        cleaned = re.sub(r'Go back to Summary|Case Rubric continued', '', content)
        return clean_text(cleaned)
    return ""


def parse_case_study(text):
    """Main parser with enhanced section handling."""
    return {
        "patient_info": extract_patient_info(text),
        "summary": extract_summary(text),
        "team_members": extract_team_members(text),
        "background": extract_background(text),
        "assessment_plan": parse_assessment_plan(text),
        "assessment_results": parse_assessment_results(text),
        "treatment_plan": parse_treatment_plan(text)
    }


def process_zip(zip_path, output_dir="Case_Study_json"):
    """Process zip file with improved error handling."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            if file_info.filename.lower().endswith('.txt'):
                with zip_ref.open(file_info) as file:
                    text_content = file.read().decode('utf-8')
                    json_data = parse_case_study(text_content)

                    json_filename = Path(file_info.filename).stem + '.json'
                    output_path = Path(output_dir) / json_filename

                    with open(output_path, 'w', encoding='utf-8') as json_file:
                        json.dump(json_data, json_file, indent=2, ensure_ascii=False)


print(os.getcwd())
os.chdir("../../../Case_Study_Data")

input_zip = "case-study-text.zip"  # Replace with your zip file path
process_zip(input_zip)
print(f"Processed all files in {input_zip} and saved JSONs to Case_Study_json directory")