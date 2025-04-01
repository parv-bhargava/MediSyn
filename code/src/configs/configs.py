PROMPT_GENERATOR_PROMPT="Generate a prompt for this role as a {}."
SYNTHESIS_AGENT_PROMPT="""As the synthesis agent, your task is to carefully analyze and integrate the recommendations provided by each role-specific healthcare agent involved in the current clinical case. Each agent's input represents their specialized professional perspective.
Your goal is to synthesize these diverse insights into a cohesive, context-sensitive, and comprehensive treatment plan.

Follow these guidelines strictly:

1. Read all the individual recommendations thoroughly to understand each professional perspective.
2. Identify areas of agreement, differences, or potential conflicts among the recommendations.
3. Ensure that the final synthesized treatment plan respects the contributions from each professional perspective.
4. Clearly outline specific, actionable recommendations under each team member’s role.

Use EXACTLY the following format for your final synthesized response:

Required Output Format
Create a comprehensive treatment plan with specific recommendations from each team member's perspective.

Structure your response using EXACTLY these section headers:
{}
{{For each team member}}
[Full Team Member Role Name]: [Recommendations]

Ensure your synthesis is accurate, nuanced, and demonstrates an integrated inter-professional approach that aligns with Inter-Professional Education/Practice (IPE/IPP) principles.
"""

JUDGE_PROMPT = """
Evaluate On the Basis of the Following Criteria:
The diagnostic accuracy framework was categorized into the following components: 
(1) Overall Accuracy, 
(2) Plausibility, 
(3) Specificity, and 
(4) Omission/ Uncertainty. 

Accuracy, Plausibility, and Specificity were applied at the individual diagnosis level and were conditioned such that only diagnoses classified as accurate were scored for plausibility and only those classified as plausible were scored for specificity. Accuracy represented how well it met the definition of a diagnosis as stated above. Plausibility represented if the diagnosis was hallucinated and could pose potential harm. Finally, Specificity captured the level of detail in the diagnosis (i.e., sepsis vs sepsis from influenza pneumonia). The final component of diagnostic evaluation, Omission/ Uncertainty, was applied to the entire list of outputted diagnoses. Omission captured instances in which a diagnosis was missing from the output, but would be considered in a clinical setting. Uncertainty, which is conditional on Omission, further penalized a model for not utilizing the information it was provided versus not being provided with enough information.

"""

BASE_PROMPT = """Generate a treatment plan."""

TUNED_PROMPT = """
**Required Output Format**
Create a comprehensive treatment plan with specific recommendations from each team member's perspective.

Structure your response using EXACTLY these section headers:
{}
{{For each team member}}
[Full Team Member Role Name]: [Recommendations]
"""





