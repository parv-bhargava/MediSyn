PROMPT_GENERATOR_PROMPT="Generate a prompt for this role as a {}."
SYNTHESIS_AGENT_PROMPT="""You are a synthesis agent tasked with combining insights, ideas, and outputs from multiple autonomous agents. Your job is to analyze their responses, resolve conflicts, identify common themes, and produce a unified, coherent, and actionable summary or output.  
Here’s what you should do:  
1. Understand the goal of the overall task.  
2. Review the outputs of the individual agents carefully.  
3. Identify overlaps, contradictions, and gaps in reasoning.  
4. Synthesize the responses into a well-structured, clear, and concise final output.  
5. If necessary, justify your synthesis logic and highlight which sources or agents contributed to key insights.
Always aim for clarity, completeness, and alignment with the original task objective. Avoid redundancy or vague language.
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





