# MediSyn: A Synergistic Multi-Agent AI System for IPE in Medical Domain

## Overview
MediSyn is a novel multi-agent AI system designed to simulate interprofessional collaboration in healthcare settings. The system dynamically creates role-specific agents based on the healthcare team members mentioned in medical case studies, with each agent providing recommendations from their professional perspective. A synthesis agent then combines these recommendations into a cohesive treatment plan.

This project explores how a specialized multi-agent design compares to more general-purpose strategies like single LLM and prompt-tuned LLM approaches in generating comprehensive treatment plans for interprofessional education (IPE) case studies.

## Key Features
- **Dynamic Agent Creation**: Automatically creates specialized agents based on healthcare roles identified in case studies
- **Role-Specific Reasoning**: Each agent is constrained to reason within its professional domain
- **Collaborative Synthesis**: Integrates diverse perspectives into a cohesive treatment plan
- **Model Agnostic**: Compatible with multiple LLM backends (GPT-4o, Claude 3.5, LLaMA 3.2)
- **Comprehensive Evaluation**: Includes metrics for both automated (BLEU, ROUGE) and qualitative (accuracy, plausibility, specificity, omission) assessment

## Project Structure
```
code/
├── src/                      # Source code
│   ├── agents/               # Agent implementation
│   │   ├── base_agent.py     # Base agent class
│   │   ├── dynamic_agent.py  # Dynamic agent manager
│   ├── configs/              # Configuration files
│   │   ├── configs.py        # System prompts and configurations
│   ├── data/                 # Case study data
│   │   ├── casestudy/        # Manually prepared case studies
│   ├── eval/                 # Evaluation modules
│   │   ├── evaluate.py       # Treatment plan evaluator
│   ├── main.py               # Main execution script
```

## Installation

### Prerequisites
- Python 3.13
- AWS account with Bedrock access (for Claude and LLaMA models)
- OpenAI API key (for GPT-4o)

### Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/MediSyn.git
cd MediSyn/code
```

2. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure API keys:
   - Create a `.env` file in the `src/agents/` directory with the following content:
   ```
   OPENAI_API_KEY=your_openai_api_key
   aws_access_key_id=your_aws_access_key
   aws_secret_access_key=your_aws_secret_key
   aws_session_token=your_aws_session_token 
   ```

## Usage

### Running the System
To run the MediSyn system on case studies:

```bash
cd src
python main.py
```

By default, this will:
1. Process case studies from the `data/casestudy` directory
2. Generate treatment plans using base, prompt-tuned, and MediSyn approaches
3. Save outputs to the `data/outputs` directory
4. Evaluate the generated plans and save metrics to an Excel file

### Customizing Execution
You can modify the `main.py` file to:
- Change the input data directory
- Select specific LLM models
- Enable/disable evaluation
- Adjust output directories

## Evaluation
MediSyn includes a comprehensive evaluation framework that assesses treatment plans using:

1. **Automated Metrics**:
   - BLEU score: Measures n-gram overlap with reference plans
   - ROUGE scores: Assesses recall-oriented text similarity

2. **LLM-as-Judge Evaluation**:
   - Overall accuracy: Correctness and clinical relevance
   - Plausibility: Medical soundness and freedom from hallucinations
   - Specificity: Level of detail in recommendations
   - Omission: Penalization for missing crucial clinical actions

This will execute all test cases, including AWS Bedrock integration tests.

## Contributors
- Sajan K. Kar (sajan.kar@gwu.edu)
- Parv Bhargava (parv.bhargava@gwu.edu)
- Amir Jafari (ajafari@gwu.edu)

## Acknowledgments
This project uses case studies from the American Speech-Language-Hearing Association (ASHA) IPE/IPP portal.