import json
import re

def parse_case_study(text):
    data = {
        "patient_info": {},
        "summary": "",
        "team_members": [],
        "background": "",
        "assessment_plan": {
            "general_plan": "",
            "Neonatologist/NNP": "",
            "Pediatric otolaryngologist": "",
            "NICU nurse": "",
            "Lactation consultant": "",
            "SLP": "",
            "Physical therapy": "",
            "Family": ""
        },
        "assessment_results": {"general_results": "", "roles": {}},
        "treatment_plan": {"general_plan": "", "roles": {}}
    }

    def clean_text(t):
        return ' '.join(t.split()).strip()

    # Patient Info
    patient_match = re.search(r'Meet The Team\n\n(.+?)\n\n', text, re.DOTALL)
    if patient_match:
        patient_line = patient_match.group(1).strip().split('\n')[0]
        if patient_line:
            # Split on first space only
            name_age = patient_line.split(' ', 1)  # Split into [name, age]
            data["patient_info"] = {
                "name": name_age[0].strip(),
                "age": name_age[1].strip() if len(name_age) > 1 else "",
                "diagnosis": []
            }
            diagnosis_match = re.search(r'Current Diagnosis: (.+?)\n', text)
            if diagnosis_match:
                data["patient_info"]["diagnosis"] = [d.strip() for d in diagnosis_match.group(1).split(',')]

    # Summary
    summary_match = re.search(r'Summary\n\n(.+?)\nPatient Info', text, re.DOTALL)
    if summary_match:
        data["summary"] = clean_text(summary_match.group(1))

    # Team Members
    team_section_match = re.search(r'Meet The Team\n\n(.+?)\nFamily', text, re.DOTALL)
    if team_section_match:
        lines = [line.strip() for line in team_section_match.group(1).split('\n') if line.strip()]
        valid_members = [line for line in lines[1:] if not line.startswith("Current Diagnosis:")]
        data["team_members"] = list(dict.fromkeys(valid_members))

    # Background
    background_match = re.search(r'Background\n\n(.+?)\nHow They Collaborated', text, re.DOTALL)
    if background_match:
        data["background"] = clean_text(background_match.group(1))

    # Assessment Plan
    assessment_plan_match = re.search(
        r'Assessment Plan \(Determine roles/ responsibilities for evaluation\)\n\n(.+?)\nContinue for more',
        text, re.DOTALL
    )
    if assessment_plan_match:
        content = re.sub(r'Go back to Summary\n?', '', assessment_plan_match.group(1))
        parts = re.split(r'\n(?=\w+.*?:)', content)
        if parts:
            data["assessment_plan"]["general_plan"] = clean_text(parts[0])
            for part in parts[1:]:
                role_desc = re.match(r'(.+?):\s*(.+?)(?=\n\w+.*?:|$)', part, re.DOTALL)
                if role_desc:
                    role = role_desc.group(1).strip()
                    description = clean_text(role_desc.group(2))
                    if role in data["assessment_plan"]:
                        data["assessment_plan"][role] = description

    # Section parser with dynamic general key
    def parse_section(pattern, section, general_key):
        match = re.search(pattern, text, re.DOTALL)
        if match:
            content = re.sub(r'Go back to Summary\n?', '', match.group(1))
            parts = re.split(r'\n(?=\w+.*?:)', content)
            if parts:
                section[general_key] = clean_text(parts[0])
                for part in parts[1:]:
                    role_desc = re.match(r'(.+?):\s*(.+?)(?=\n\w+.*?:|$)', part, re.DOTALL)
                    if role_desc:
                        section["roles"][role_desc.group(1).strip()] = clean_text(role_desc.group(2))

    # Assessment Results with general_results
    parse_section(
        r'Assessment Results \(Summarize key diagnostic results\)\n\n(.+?)\nContinue for more',
        data["assessment_results"],
        "general_results"
    )

    # Treatment Plan with general_plan
    parse_section(
        r'IPP Treatment Plan \(Discuss, reflect, and modify recommendations to develop a coordinated plan\)\n\n(.+?)\nContinue for more',
        data["treatment_plan"],
        "general_plan"
    )

    return data

# Usage
with open('case-study-20.txt', 'r', encoding='utf-8') as f:
    text = f.read()

result = parse_case_study(text)

with open('output.json', 'w') as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))