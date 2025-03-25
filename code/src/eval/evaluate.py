import json
import os
import re
import nltk
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

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

                generated, reference = self.load_data(output_path, ref_path)
                bleu = self.calculate_bleu(generated, reference)
                rouge = self.calculate_rouge(generated, reference)

                results.append({
                    "output_file": fname,
                    "case_study": reference_name,
                    "bleu": bleu,
                    "rouge1": rouge['rouge1'].fmeasure,
                    "rouge2": rouge['rouge2'].fmeasure,
                    "rougeL": rouge['rougeL'].fmeasure
                })


        df = pd.DataFrame(results)
        return df


if __name__ == '__main__':
    print("Current working directory:", os.getcwd())
    os.chdir("../data")
    print("Changed working directory to:", os.getcwd())

    evaluator = TreatmentEvaluator()
    df_metrics = evaluator.evaluate(output_dir="outputs", reference_dir=".")

    print("\nEvaluation Results DataFrame:")
    print(df_metrics)
