import json
import os
import re

from agents.dynamic_agent import DynamicAgentManager
from eval.evaluate import TreatmentEvaluator
from agents.base_agent import Agent
from configs.configs import BASE_PROMPT, TUNED_PROMPT


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


def save_output(case_study_path: str, generated_text: str, tag: str = "", output_dir="data/outputs"):
    """
    Save generated output to a JSON file with a case-specific and tag-specific filename.
    Sanitizes file names to remove illegal characters like colons (:) and slashes.
    """

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
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    return output_path


def main(eval=False, case_study_dir="data/Case_Study_Json_Manual", output_dir="data/outputs"):
    model_ids = [
        # "meta.llama3-70b-instruct-v1:0",
        # "anthropic.claude-3-5-sonnet-20240620-v1:0"
        # "gpt-4o"
        "hf:meta-llama/Llama-3.3-70B-Instruct"
    ]

    print(output_dir)
    for fname in os.listdir(case_study_dir):
        if fname.endswith(".json"):
            case_study_path = os.path.join(case_study_dir, fname)
            case_study, team_list = generate_prompt(case_study_path)

            # --- Base Agent responses ---
            for model_id in model_ids:
                agent = Agent(
                    name="Base Agent",
                    role=BASE_PROMPT,
                    input=case_study,
                    model_id=model_id
                )
                response = agent.run()
                print(f"Base Agent Response for model {model_id} on {fname}:\n{response}\n")
                save_output(case_study_path, response, tag=f"{model_id}-base", output_dir=output_dir)

            # --- Tuned Agent responses ---
            for model_id in model_ids:
                agent = Agent(
                    name="Tuned Agent",
                    role=TUNED_PROMPT.format(team_list),
                    input=case_study,
                    model_id=model_id
                )
                response = agent.run()
                print(f"Tuned Agent Response for model {model_id} on {fname}:\n{response}\n")
                save_output(case_study_path, response, tag=f"{model_id}-tuned", output_dir=output_dir)

            # --- Multi-Agent (Synthesis) responses ---
            for model_id in model_ids:
                mas = DynamicAgentManager(model_id=model_id)
                mas.run_manager(team_list, "Prepare a treatment plan for this case study." + case_study, synthesis=True)

                print(f"\nStored Agent Responses for model {model_id} on {fname}:")
                for agent_name, response in mas.memory.items():
                    print(f"Agent ID: {agent_name}\nResponse:\n{response}\n")
                    if agent_name == "SYNTHESIS_AGENT":
                        save_output(case_study_path, response, tag=f"{model_id}-synthesis", output_dir=output_dir)
    if eval:
        evaluator = TreatmentEvaluator()
        df_metrics = evaluator.evaluate(
            output_dir=output_dir,
            reference_dir=case_study_dir
        )

        eval_filename = f"evaluation_results_{model_ids}.xlsx"
        eval_filepath = os.path.join(output_dir, eval_filename)
        df_metrics.to_excel(eval_filepath, index=False)
        print("\nEvaluation Results DataFrame:")
        print(df_metrics)


if __name__ == "__main__":
    # Base main
    # main(eval=True, output_dir="data/outputs_manual_llama")
    # main(eval=True, output_dir="data/outputs_manual_claude")
    # main(eval=True, output_dir="data/outputs_manual_gpt4o")
    main(eval=True, output_dir="data/outputs_manual_hf_llama")
    # Claude main
    # main(eval=True, case_study_dir="data/Claude_case_study_json", output_dir="data/outputs_claude")
    # Llama main
    # main(eval=True, case_study_dir="data/Llama_case_study_json", output_dir="data/outputs_llama")
    # gpt-4o main
    # main(eval=True, case_study_dir="data/gpt_case_study_json", output_dir="data/outputs_gpt4o")
    # main(eval=True, case_study_dir="data/Llama_case_study_json", output_dir="data/outputs_manual_hf_llama")
