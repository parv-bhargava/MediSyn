import json
import re
import os


def clean_text(t):
    return ' '.join(t.split()).strip()


def extract_patient_info(text):
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
    summary_match = re.search(r'Summary\n\n(.+?)\nPatient Info', text, re.DOTALL)
    return clean_text(summary_match.group(1)) if summary_match else ""


def extract_team_members(text):
    team_members = []
    team_section_match = re.search(r'Meet The Team\n\n(.+?)\nFamily', text, re.DOTALL)
    if team_section_match:
        lines = [line.strip() for line in team_section_match.group(1).split('\n') if line.strip()]
        valid_members = [line for line in lines[1:] if not line.startswith("Current Diagnosis:")]
        team_members = list(dict.fromkeys(valid_members))
    return team_members


def extract_background(text):
    background_match = re.search(r'Background\n\n(.+?)\nHow They Collaborated', text, re.DOTALL)
    return clean_text(background_match.group(1)) if background_match else ""


def parse_assessment_results(text):
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

def parse_section(text, start_pattern, end_pattern, general_key):
    section = {general_key: "", "roles": {}}
    # Use more flexible section boundary detection
    section_match = re.search(
        fr'{start_pattern}\n\n(.*?)(?=\n{end_pattern}|\Z)',
        text,
        re.DOTALL
    )
    if not section_match:
        return section

    content = section_match.group(1)
    # Enhanced cleaning of page navigation elements
    cleaned_content = re.sub(
        r'Go back to Summary|Continue for more|Summary Page \d+ of \d+|\*+ Case Rubric \d+ of \d+',
        '',
        content
    )
    # Improved role splitting with lookahead for role patterns
    parts = re.split(r'\n(?=\s*\b[\w\s/]+:)', cleaned_content)

    if parts:
        # Process general text
        section[general_key] = clean_text(parts[0])

        # Process roles with enhanced regex
        for part in parts[1:]:
            # Handle multi-line descriptions and special characters
            role_match = re.match(r'^\s*([\w\s/]+):\s*(.+?)(?=\n\s*[\w\s/]+:|\Z)', part, re.DOTALL)
            if role_match:
                role = role_match.group(1).strip()
                description = clean_text(role_match.group(2))
                section["roles"][role] = description

    return section


def parse_case_study(text):
    return {
        "patient_info": extract_patient_info(text),
        "summary": extract_summary(text),
        "team_members": extract_team_members(text),
        "background": extract_background(text),
        "assessment_plan": parse_assessment_plan(text),
        "assessment_results": parse_assessment_results(text),
        "treatment_plan": parse_treatment_plan(text)  # Use new function
    }


# Usage remains the same
print(os.getcwd())
os.chdir("../../../Case_Study_Data")

with open('case-study-20.txt', 'r', encoding='utf-8') as f:
    text = f.read()

result = parse_case_study(text)

with open('case-study-20.json', 'w') as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))