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
claude_df = pd.read_excel("outputs_claude/evaluation_results_['anthropic.claude-3-5-sonnet-20240620-v1:0'].xlsx")
gpt_df = pd.read_excel("outputs_gpt4o/evaluation_results_['gpt-4o'].xlsx")
llama_df = pd.read_excel("outputs_llama/evaluation_results_['meta.llama3-70b-instruct-v1:0'].xlsx")

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


def bar_plots(summary, llm_name, data):
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        axes[i].bar(summary['method'], summary[metric])
        axes[i].set_title(metric)
        axes[i].set_ylabel("Average Score")
        axes[i].set_xlabel("Method")
    fig.suptitle(f'{llm_name} Evaluation ({data})')
    plt.tight_layout()
    plt.savefig(f"{llm_name}_{data.replace(' ', '_')}_barplot.pdf", format='pdf')
    plt.close()


# Bar plots for average evaluation metrics by method(base vs tuned vs synthesis) for manually prepared data
manual_claude_df['method'] = manual_claude_df['output_file'].apply(extract_method)
claude_summary = manual_claude_df.groupby('method')[metrics].mean().reset_index()

manual_gpt_df['method'] = manual_gpt_df['output_file'].apply(extract_method)
gpt_summary = manual_gpt_df.groupby('method')[metrics].mean().reset_index()

manual_llama_df['method'] = manual_llama_df['output_file'].apply(extract_method)
llama_summary = manual_llama_df.groupby('method')[metrics].mean().reset_index()

bar_plots(claude_summary, llm_name="Claude", data='Data Prepared Manually')
bar_plots(gpt_summary, llm_name="GPT-4o", data='Data Prepared Manually')
bar_plots(llama_summary, llm_name="Llama", data='Data Prepared Manually')

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
# final_table.to_csv('../eval/Final_eval_DPM.csv')

# -------------------------------------- XXXX ---------------------------------------

# Line chart showing metric trends across models per method

# Average per metric per method per model for line chart
line_data = pd.concat([
    base_summary.assign(method='base'),
    tuned_summary.assign(method='tuned'),
    synthesis_summary.assign(method='synthesis')
])

# Melt the dataframe to long format for plotting
line_df = line_data.melt(id_vars=['model', 'method'], var_name='metric', value_name='score')

# Plot
plt.figure(figsize=(12, 6))
for model in line_df['model'].unique():
    subset = line_df[(line_df['model'] == model) & (line_df['metric'] == 'overall_accuracy')]
    plt.plot(subset['method'], subset['score'], label=model, marker='o')

plt.title('Overall Accuracy Comparison Across Methods')
plt.xlabel('Method')
plt.ylabel('Average Overall Accuracy')
plt.legend(title='Model')
plt.grid(True)
plt.tight_layout()
plt.savefig("Overall_Accuracy_Comparison_Manual.pdf", format='pdf')
plt.close()

# -------------------------------------- XXXX ---------------------------------------

# Bar plots for average evaluation metrics by method(base vs tuned vs synthesis) for LLM prepared data
claude_df['method'] = claude_df['output_file'].apply(extract_method)
claude_summary = claude_df.groupby('method')[metrics].mean().reset_index()

gpt_df['method'] = gpt_df['output_file'].apply(extract_method)
gpt_summary = gpt_df.groupby('method')[metrics].mean().reset_index()

llama_df['method'] = llama_df['output_file'].apply(extract_method)
llama_summary = llama_df.groupby('method')[metrics].mean().reset_index()

bar_plots(claude_summary, llm_name="Claude", data='Data Prepared by Claude')
bar_plots(gpt_summary, llm_name="GPT-4o", data='Data Prepared by GPT')
bar_plots(llama_summary, llm_name="Llama", data='Data Prepared Llama')

# -------------------------------------- XXXX ---------------------------------------

gpt_df['model'] = 'gpt-4o'
llama_df['model'] = 'llama3'
claude_df['model'] = 'claude-3.5'

# Combine all into a single DataFrame
combined_df = pd.concat([gpt_df, llama_df, claude_df], ignore_index=True)

base_df = combined_df[combined_df['method'] == 'base']
tuned_df = combined_df[combined_df['method'] == 'tuned']
synthesis_df = combined_df[combined_df['method'] == 'synthesis']

synthesis_summary = synthesis_df.groupby('model')[metrics].mean().reset_index()
base_summary = base_df.groupby('model')[metrics].mean().reset_index()
tuned_summary = tuned_df.groupby('model')[metrics].mean().reset_index()

final_table_dpl = reformat_summary_table(base_summary, tuned_summary, synthesis_summary)
# final_table_dpl.to_csv('../eval/Final_eval_DPL.csv')

# -------------------------------------- XXXX ---------------------------------------

# Line chart showing metric trends across models per method for LLM prepared data

# Average per metric per method per model for line chart
line_data = pd.concat([
    base_summary.assign(method='base'),
    tuned_summary.assign(method='tuned'),
    synthesis_summary.assign(method='synthesis')
])

# Melt the dataframe to long format for plotting
line_df_llm = line_data.melt(id_vars=['model', 'method'], var_name='metric', value_name='score')

# Plot
plt.figure(figsize=(12, 6))
for model in line_df_llm['model'].unique():
    subset = line_df_llm[(line_df_llm['model'] == model) & (line_df_llm['metric'] == 'overall_accuracy')]
    plt.plot(subset['method'], subset['score'], label=model, marker='o')

plt.title('Overall Accuracy Comparison Across Methods')
plt.xlabel('Method')
plt.ylabel('Average Overall Accuracy')
plt.legend(title='Model')
plt.grid(True)
plt.tight_layout()
plt.savefig("Overall_Accuracy_Comparison_LLM.pdf", format='pdf')
plt.close()