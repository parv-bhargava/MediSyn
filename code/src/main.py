import json
import os
import re

from agents.dynamic_agent import DynamicAgentManager
from eval.evaluate import TreatmentEvaluator
from agents.base_agent import Agent


def generate_prompt(case_study_path):
    """
    Reads a case study JSON and returns the formatted prompt plus team member list.
    """
    with open(case_study_path, 'r', encoding='utf-8') as f:
        case_data = json.load(f)

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
    return prompt, case_data['team_members']


def save_output(case_study_path: str, generated_text: str, tag: str = ""):
    """
    Save generated output to a JSON file with a case-specific and tag-specific filename.
    Sanitizes file names to remove illegal characters like colons (:) and slashes.
    """
    output_dir = "data/outputs"
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(case_study_path))[0]
    safe_base_name = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', base_name)
    safe_tag = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', tag)

    if safe_tag:
        output_filename = f"treatment-plan-{safe_base_name}-{safe_tag}.json"
    else:
        output_filename = f"treatment-plan-{safe_base_name}.json"

    output_path = os.path.join(output_dir, output_filename)

    output_data = {
        "source_file": case_study_path,
        "generated_text": generated_text
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    return output_path


def main():
    model_ids = [
        "meta.llama3-70b-instruct-v1:0",
        "anthropic.claude-3-5-sonnet-20240620-v1:0"
    ]

    case_study_dir = "data/casestudy"

    for fname in os.listdir(case_study_dir):
        if fname.endswith(".json"):
            case_study_path = os.path.join(case_study_dir, fname)
            case_study, team_list = generate_prompt(case_study_path)

            # --- Base Agent responses ---
            for model_id in model_ids:
                agent = Agent(
                    name="Base Agent",
                    role="Provide accurate, evidence-based recommendations. Always prioritize patient safety.",
                    conversation=case_study,
                    model_id=model_id
                )
                response = agent.run()
                print(f"Base Agent Response for model {model_id} on {fname}:\n{response}\n")
                save_output(case_study_path, response, tag=f"{model_id}-base")

            # --- Multi-Agent (Synthesis) responses ---
            for model_id in model_ids:
                mas = DynamicAgentManager(model_id=model_id)
                mas.run_manager(team_list, "Prepare a treatment plan for this case study." + case_study, synthesis=True)

                print(f"\nStored Agent Responses for model {model_id} on {fname}:")
                for agent_name, response in mas.memory.items():
                    print(f"Agent ID: {agent_name}\nResponse:\n{response}\n")
                    if agent_name == "SYNTHESIS_AGENT":
                        save_output(case_study_path, response, tag=f"{model_id}-synthesis")

    evaluator = TreatmentEvaluator()
    df_metrics = evaluator.evaluate(
        output_dir="data/outputs",
        reference_dir="data/casestudy"
    )

    df_metrics.to_excel("evaluation_results.xlsx", index=False)
    print("\nEvaluation Results DataFrame:")
    print(df_metrics)


if __name__ == "__main__":
    main()
