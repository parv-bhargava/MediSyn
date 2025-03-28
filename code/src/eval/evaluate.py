import json
import os
import re
import nltk
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from agents.base_agent import Agent
import ast
from configs.configs import JUDGE_PROMPT
from pydantic import BaseModel
from rouge_score import rouge_scorer

class DAFModel(BaseModel):
    """
    Diagnostic Accuracy Framework Model:
    Link:https://pmc.ncbi.nlm.nih.gov/articles/PMC10984060/#:~:text=The%20human%20evaluation%20was%20split,be%20made%20through%20physical%20examination
    """
    reasoning: str
    overall_accuracy: float
    plausibility  : float
    specificity  : float
    omission : float


class TreatmentEvaluator:
    def __init__(self):
        # nltk.download('punkt', quiet=True)
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
        self.smoother = SmoothingFunction().method4

    def load_data(self, output_path, reference_path):
        """Load generated and reference texts"""
        with open(output_path, 'r', encoding='utf-8') as f:
            generated = json.load(f)['generated_text']

        with open(reference_path, 'r', encoding='utf-8') as f:
            reference = json.load(f)['treatment_plan']

        return generated, reference

    def calculate_bleu(self, generated, reference):
        """Calculate BLEU score with smoothing"""
        gen_tokens = nltk.word_tokenize(generated.lower())
        ref_tokens = nltk.word_tokenize(reference.lower())
        return sentence_bleu([ref_tokens], gen_tokens, smoothing_function=self.smoother)

    def calculate_rouge(self, generated, reference):
        """Calculate ROUGE scores"""
        return self.scorer.score(reference, generated)

    def llm_as_judge(self, generated, reference, model_id):
        """Calculate LLM as judgement"""
        combined = f"{reference} {generated}"
        return Agent(name="JUDGE", role=JUDGE_PROMPT, input=combined,
                     model_id=model_id).structure_run(DAFModel)

    def evaluate(self, output_dir="outputs", reference_dir="."):
        """
        Evaluate all outputs in directory and return a DataFrame with the results.
        We locate the reference file by searching for a pattern 'case-study-<number>'
        in the output filename, then appending '.json'.
        """
        results = []

        for fname in os.listdir(output_dir):
            if fname.startswith('treatment-plan-'):
                output_path = os.path.join(output_dir, fname)
                match = re.search(r'(case-study-\d+)', fname)
                if not match:
                    print(f"Warning: Could not find a case-study pattern in {fname}")
                    continue

                reference_name = match.group(1) + ".json"
                ref_path = os.path.join(reference_dir, reference_name)

                if not os.path.exists(ref_path):
                    print(f"Warning: No reference found for {fname}")
                    continue
                model_id = 'gpt-4o'
                generated, reference = self.load_data(output_path, ref_path)
                bleu = self.calculate_bleu(generated, reference)
                rouge = self.calculate_rouge(generated, reference)
                #According to diagnostic accuracy framework https://pmc.ncbi.nlm.nih.gov/articles/PMC10984060/#:~:text=The%20human%20evaluation%20was%20split,be%20made%20through%20physical%20examination
                daf = self.llm_as_judge(generated, reference, model_id)

                results.append({
                    "output_file": fname,
                    "case_study": reference_name,
                    "bleu": bleu,
                    "rouge1": rouge['rouge1'].fmeasure,
                    "rouge2": rouge['rouge2'].fmeasure,
                    "rougeL": rouge['rougeL'].fmeasure,
                    "overall_accuracy": daf.overall_accuracy,
                    "plausibility": daf.plausibility,
                    "specificity": daf.specificity,
                    "omission": daf.omission
                })
                print(results)

        df = pd.DataFrame(results)
        return df


if __name__ == '__main__':
    print("Current working directory:", os.getcwd())
    os.chdir("../data")
    print("Changed working directory to:", os.getcwd())

    evaluator = TreatmentEvaluator()
    df_metrics = evaluator.evaluate(output_dir="outputs", reference_dir="casestudy")
    df_metrics.to_excel("evaluation_results.xlsx")
