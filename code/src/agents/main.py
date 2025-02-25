import json
import os
from dynamic_agent import DynamicAgentManager


def generate_prompt(case_study_path):
    with open(case_study_path, 'r', encoding='utf-8') as f:
        case_data = json.load(f)
    team_list = '\n'.join([f"- {role}" for role in case_data['team_members']])
    prompt = f"""**IPE Case Study Analysis Task**

            **Patient Information**
            Name: {case_data['patient_info']['name']}
            Age: {case_data['patient_info']['age']}
            Diagnoses: {', '.join(case_data['patient_info']['diagnosis'])}

            **Case Summary**
            {case_data['summary']}

            **Medical Background**
            {case_data['background']}

            **Assessment Plan**
            {case_data['assessment_plan']}

            **Assessment Results**
            {case_data['assessment_results']}

            """
    return prompt, team_list


def save_output(case_study_path: str, generated_text: str):
    """Save generated output with case-specific filename"""
    output_dir = "../data/outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Extract base filename and create output name
    base_name = os.path.splitext(os.path.basename(case_study_path))[0]
    output_filename = f"treatment-plan-{base_name}.json"
    output_path = os.path.join(output_dir, output_filename)

    output_data = {
        "source_file": case_study_path,
        "generated_text": generated_text
    }
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    return output_path


def main():
    case_study, team_list = generate_prompt("../data/case-study-25.json")
    # TODO: Create regex for team_list
    roles = ["physician", "oncologist", "nurse"]
    base_prompt = "Prepare a treatment plan for this case study. " + case_study

    manager = DynamicAgentManager()
    manager.run_manager(roles, base_prompt, synthesis=True)

    print("\nStored Agent Responses:")
    for agent_name, response in manager.memory.items():
        print(f"Agent ID: {agent_name}\nResponse: {response}\n")
        if agent_name == "SYNTHESIS_AGENT":
            save_output("../data/case-study-25.json", response)
if __name__ == "__main__":
    main()
