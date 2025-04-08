import pandas as pd
import matplotlib.pyplot as plt
import os
import re
print(os.getcwd())
os.chdir('../data/')
print(os.getcwd())

# Read the Excel files
manual_claude_df=pd.read_excel("outputs_manual_claude/evaluation_results_['anthropic.claude-3-5-sonnet-20240620-v1:0'].xlsx")
manual_gpt_df = pd.read_excel("outputs_manual_gpt4o/evaluation_results_['gpt-4o'].xlsx")
manual_llama_df = pd.read_excel("outputs_manual_llama/evaluation_results_['meta.llama3-70b-instruct-v1:0'].xlsx")

metrics = [
    'bleu', 'rouge1', 'rouge2', 'rougeL',
    'overall_accuracy', 'plausibility', 'specificity', 'omission'
]
def extract_method(filename):
    if 'base' in filename:
        return 'base'
    elif 'tuned' in filename:
        return 'tuned'
    elif 'synthesis' in filename:
        return 'synthesis'
    else:
        return 'unknown'


def bar_plots(summary, llm_name):
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        axes[i].bar(summary['method'], summary[metric])
        axes[i].set_title(metric)
        axes[i].set_ylabel("Average Score")
        axes[i].set_xlabel("Method")
    fig.suptitle(f'{llm_name} Evaluation (Data Prepared Manually)')
    plt.tight_layout()
    plt.show()


manual_claude_df['method'] = manual_claude_df['output_file'].apply(extract_method)
claude_summary = manual_claude_df.groupby('method')[metrics].mean().reset_index()

manual_gpt_df['method'] = manual_gpt_df['output_file'].apply(extract_method)
gpt_summary = manual_gpt_df.groupby('method')[metrics].mean().reset_index()

manual_llama_df['method'] = manual_llama_df['output_file'].apply(extract_method)
llama_summary = manual_llama_df.groupby('method')[metrics].mean().reset_index()

bar_plots(claude_summary, llm_name="Claude")
bar_plots(gpt_summary, llm_name="GPT-4o")
bar_plots(llama_summary, llm_name="Llama")

# -------------------------------------- XXXX ---------------------------------------

def reformat_summary_table(base, tuned, synthesis):
    # Rename columns for multi-index look
    base = base.set_index('model').T
    tuned = tuned.set_index('model').T
    synthesis = synthesis.set_index('model').T

    # Create MultiIndex columns: (method, model)
    base.columns = pd.MultiIndex.from_product([['Base Agent'], base.columns])
    tuned.columns = pd.MultiIndex.from_product([['Prompt Tuned'], tuned.columns])
    synthesis.columns = pd.MultiIndex.from_product([['Synthesis-Agent'], synthesis.columns])

    # Concatenate along columns
    combined = pd.concat([base, tuned, synthesis], axis=1)

    # Reorder rows as per user's desired order
    ordered_rows = ['bleu', 'rouge1', 'rouge2', 'rougeL',
                    'overall_accuracy', 'plausibility', 'specificity', 'omission']
    combined = combined.loc[ordered_rows]
    combined.index = ['BLEU', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L',
                      'Accuracy', 'Plausibility', 'Specificity', 'Omission']
    return combined


manual_gpt_df['model'] = 'gpt-4o'
manual_llama_df['model'] = 'llama3'
manual_claude_df['model'] = 'claude-3.5'

# Combine all into a single DataFrame
combined_df = pd.concat([manual_gpt_df, manual_llama_df, manual_claude_df], ignore_index=True)

base_df = combined_df[combined_df['method'] == 'base']
tuned_df = combined_df[combined_df['method'] == 'tuned']
synthesis_df = combined_df[combined_df['method'] == 'synthesis']

synthesis_summary = synthesis_df.groupby('model')[metrics].mean().reset_index()
base_summary = base_df.groupby('model')[metrics].mean().reset_index()
tuned_summary = tuned_df.groupby('model')[metrics].mean().reset_index()

final_table = reformat_summary_table(base_summary, tuned_summary, synthesis_summary)
final_table.to_csv('../eval/Final_eval_DMP.csv')