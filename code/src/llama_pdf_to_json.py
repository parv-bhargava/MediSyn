import json
from json import JSONDecodeError
import os
import boto3
from unstructured.partition.pdf import partition_pdf
from dotenv import load_dotenv
import re
import glob


# Your existing LLM class
class LLM:
    """
    LLM class to interact with Bedrock Models.
    """

    def __init__(self, region_name="us-east-1"):
        self.region_name = region_name
        self.aws_access_key_id = None
        self.aws_secret_access_key = None
        self.aws_session_token = None
        self.client = None
        self._load_credentials()

    def _load_credentials(self):
        """Load AWS credentials from environment variables."""
        load_dotenv()
        self.aws_access_key_id = os.getenv("aws_access_key_id")
        self.aws_secret_access_key = os.getenv("aws_secret_access_key")
        self.aws_session_token = os.getenv("aws_session_token")

    def create_client(self):
        """Create and return a Bedrock client."""
        if not all([self.aws_access_key_id, self.aws_secret_access_key]):
            raise ValueError("AWS credentials are not properly set in the environment.")

        session = boto3.Session(
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token
        )
        self.client = session.client("bedrock-runtime", region_name=self.region_name)
        return self.client


# New PDF processing functionality
class PDFProcessor:
    def __init__(self):
        self.llm = LLM()
        self.client = self.llm.create_client()

    def extract_pdf_text(self, pdf_path):
        """Extract structured text using Unstructured library"""
        elements = partition_pdf(
            filename=pdf_path,
            strategy="fast",
            infer_table_structure=True,
            include_page_breaks=False,
        )
        return "\n\n".join([str(el) for el in elements])

    def generate_prompt(self, text_content):
        """Llama 3 optimized prompt formatting"""
        return f"""<|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
    You are a medical document processor. Extract ALL information into this EXACT JSON format:
    {{
      "patient_info": {{
        "name": "string", 
        "age": "string",
        "diagnosis": ["string"]
      }},
      "summary": "Concise clinical summary",
      "team_members": ["string"],
      "background": "Medical history and key events",
      "assessment_plan": "The assessment plan section including the individual team member assessments",
      "assessment_results": "The assessment results section including the individual team member assessments results",
      "treatment_plan": "The Treatment plan section including the individual team member's take on the treatment plan"
    }}

    RULES:
    1. Output ONLY valid JSON
    2. Use exact field names
    3. Never add comments
    4. Maintain original medical terms
    5. Be explanatory and don't summarize the information<|end_header_id|>
    <|start_header_id|>user<|end_header_id|>
    PDF CONTENT:
    {text_content}
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """

    def process_pdf(self, pdf_path):
        """Robust processing with detailed logging"""
        raw_output = ""
        try:
            print(f"\nProcessing {pdf_path}...")

            # Step 1: Extract text
            raw_text = self.extract_pdf_text(pdf_path)
            if not raw_text:
                raise ValueError("Empty PDF text extraction")
            print(f"Extracted {len(raw_text)} characters")

            # Step 2: Generate prompt
            prompt = self.generate_prompt(raw_text)
            print("Prompt generated successfully")

            # Step 3: Call Bedrock
            response = self.client.invoke_model(
                modelId="meta.llama3-70b-instruct-v1:0",
                body=json.dumps({
                    "prompt": prompt,
                    "max_gen_len": 4000,
                    "temperature": 0.3,
                })
            )
            print("Bedrock API call successful")

            # Step 4: Process response
            response_body = json.loads(response['body'].read())
            raw_output = response_body.get('generation', '')
            print(f"Raw model response:\n{raw_output}")

            # Step 5: Clean and validate
            json_str = re.search(r'\{.*\}', raw_output, re.DOTALL)
            if not json_str:
                raise ValueError("No JSON found in response")

            parsed = json.loads(json_str.group())
            print("Successfully parsed JSON")
            return parsed

        except (JSONDecodeError, ValueError) as e:
            print(f"Processing Error: {str(e)}")
            if raw_output:
                print(f"Problematic response:\n{raw_output}")
            return None
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return None

    def clean_json_output(self, raw_output):
        """Extract JSON from model output"""
        # Remove markdown code blocks
        cleaned = re.sub(r'```json|```', '', raw_output)

        # Find first { and last } to handle extra text
        start_idx = cleaned.find('{')
        end_idx = cleaned.rfind('}') + 1

        if start_idx == -1 or end_idx == 0:
            return ""

        return cleaned[start_idx:end_idx]

    def process_batch(self, file_list):
        """Process multiple PDF files"""
        results = {}
        for pdf_path in file_list:
            output = self.process_pdf(pdf_path)
            if output:
                results[pdf_path] = output
        return results

    def save_to_file(self, data, output_path):
        """Save processed data to JSON file"""
        with open(output_path, 'w',  encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved results to {output_path}")


# Usage example
if __name__ == "__main__":
    processor = PDFProcessor()

    # Set up paths
    print(os.getcwd())
    os.chdir("../../../Case_Study_Data/case-study-pdf")
    print(os.getcwd())
    input_folder = os.getcwd()
    output_folder = "../Llama_case_study_json"
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get list of PDF files
    pdf_files = glob.glob(os.path.join(input_folder, "case-study-*.pdf"))

    # Process each PDF
    for pdf_path in pdf_files:
        # Get base filename
        base_name = os.path.basename(pdf_path)
        json_name = base_name.replace(".pdf", ".json")
        output_path = os.path.join(output_folder, json_name)

        # Skip if already processed
        if os.path.exists(output_path):
            print(f"Skipping already processed file: {base_name}")
            continue

        # Process and save
        print(f"\nProcessing {base_name}...")
        result = processor.process_pdf(pdf_path)
        if result:
            processor.save_to_file(result, output_path)
        else:
            print(f"Failed to process {base_name}")

    print("\nBatch processing complete!")
