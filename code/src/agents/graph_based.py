import json
import os
from typing import TypedDict, List

from langchain_aws import BedrockLLM
from langgraph.graph import StateGraph, START, END


### Nurse ----> Physician (Also, get Nurse Insights for better results) ----> Social Worker ----> Synthesis

def generate_prompt(case_study_path: str) -> str:
    with open(case_study_path, 'r', encoding='utf-8') as f:
        case_data = json.load(f)
    team_list = '\n'.join([f"- {role}" for role in case_data['team_members']])
    prompt = f"""
        **IPE Case Study Analysis Task**

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

        **Team Members**
        {team_list}
    """
    return prompt.strip()


bedrock_llm = BedrockLLM(
    credentials_profile_name="bedrock-client",
    region_name="us-east-1",
    model_id="meta.llama3-70b-instruct-v1:0"

)

class PatientState(TypedDict):
    case_study: str
    nurse_assessment: List[str]
    physician_recommendations: List[str]
    social_worker_input: List[str]
    treatment_plan: str


def nurse_node(state: PatientState) -> dict:
    prompt = (
        f"Act as a nurse. Analyze the following patient case study and list key nursing observations "
        f"(e.g., vital signs, symptoms, social context):\n{state['case_study']}"
    )
    response = bedrock_llm.invoke(input=prompt)
    observations = [obs.strip() for obs in response.split(";") if obs.strip()]
    state["nurse_assessment"] = observations
    print("Nurse Assessment:", observations)
    return {"nurse_assessment": observations}


def physician_node(state: PatientState) -> dict:
    nurse_obs = " ".join(state.get("nurse_assessment", []))
    prompt = (
        f"Act as a physician. Based on the following nursing observations: {nurse_obs}, "
        f"provide clinical treatment recommendations for the patient."
    )
    response = bedrock_llm.invoke(input=prompt)
    recommendations = [rec.strip() for rec in response.split(";") if rec.strip()]
    state["physician_recommendations"] = recommendations
    print("Physician Recommendations:", recommendations)
    return {"physician_recommendations": recommendations}


def social_worker_node(state: PatientState) -> dict:
    prompt = (
        f"Act as a medical social worker. Review the following patient case study and suggest community support services "
        f"or interventions addressing social determinants of health:\n{state['case_study']}"
    )
    response = bedrock_llm.invoke(input=prompt)
    recommendations = [rec.strip() for rec in response.split(";") if rec.strip()]
    state["social_worker_input"] = recommendations
    print("Social Worker Input:", recommendations)
    return {"social_worker_input": recommendations}


def synthesis_node(state: PatientState) -> dict:
    nurse_obs = state.get("nurse_assessment", [])
    phys_recs = state.get("physician_recommendations", [])
    social_recs = state.get("social_worker_input", [])
    plan = (
            "Final Treatment Plan:\n"
            "• Nurse Observations: " + "; ".join(nurse_obs) + "\n"
                                                              "• Physician Recommendations: " + "; ".join(
        phys_recs) + "\n"
                     "• Social Worker Input: " + "; ".join(social_recs)
    )
    state["treatment_plan"] = plan
    print("Synthesis Output:", plan)
    return {"treatment_plan": plan}


graph = StateGraph(PatientState)
graph.add_node("nurse", nurse_node)
graph.add_node("physician", physician_node)
graph.add_node("social_worker", social_worker_node)
graph.add_node("synthesis", synthesis_node)

graph.add_edge(START, "nurse")
graph.add_edge("nurse", "physician")
graph.add_edge("physician", "social_worker")
graph.add_edge("social_worker", "synthesis")
graph.add_edge("synthesis", END)

workflow = graph.compile()
case_study_prompt = generate_prompt("../data/case-study-25.json")

initial_state: PatientState = {
    "case_study": case_study_prompt,
    "nurse_assessment": [],
    "physician_recommendations": [],
    "social_worker_input": [],
    "treatment_plan": ""
}

final_state = workflow.invoke(initial_state)
print("\nFinal Treatment Plan:\n", final_state.get("treatment_plan"))


def save_output(case_study_path: str, generated_text: str):
    """Save generated output with case-specific filename"""
    output_dir = "../data/outputs"
    os.makedirs(output_dir, exist_ok=True)

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


save_output("../data/case-study-25.json", final_state.get("treatment_plan"))
