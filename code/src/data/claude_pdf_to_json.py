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
        """Claude 3.5 optimized prompt formatting"""
        return {
            "system": """You are a medical document processor. Extract ALL information into this EXACT JSON format:
    {
      "patient_info": {
        "name": "string", 
        "age": "string",
        "diagnosis": ["string"]
      },
      "summary": "Concise clinical summary",
      "team_members": ["string"],
      "background": "Medical history and key events",
      "assessment_plan": "The assessment plan section including the individual team member assessments",
      "assessment_results": "The assessment results section including the individual team member assessments results",
      "treatment_plan": "The Treatment plan section including the individual team member's take on the treatment plan"
    }

    RULES:
    1. Output ONLY valid JSON between ```json markers
    2. Use exact field names
    3. Never add comments
    4. Maintain original medical terms
    5. Be explanatory and don't summarize the information
    6. Do not include any text outside the JSON structure""",
            "user": f"PDF CONTENT:\n{text_content}"
        }

    def process_pdf(self, pdf_path):
        """Robust processing with detailed logging and Claude 3.5 optimization"""
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

            # Step 3: Call Bedrock with Claude 3.5
            response = self.client.invoke_model(
                modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "system": prompt["system"],
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt["user"]
                                }
                            ]
                        }
                    ],
                    "max_tokens": 32000,
                    "temperature": 0.3
                })
            )
            print("Bedrock API call successful")

            # Step 4: Process response
            response_body = json.loads(response['body'].read())
            raw_output = response_body['content'][0]['text']
            print(f"Raw model response length: {len(raw_output)} characters")

            # Step 5: Clean and validate
            json_str = self.clean_json_output(raw_output)
            if not json_str:
                print("Raw response for debugging:")
                print(raw_output[:2000])  # Print first 2000 chars for inspection
                raise ValueError("No valid JSON found in response")

            parsed = json.loads(json_str)
            print("Successfully parsed JSON structure")
            return parsed

        except (JSONDecodeError, ValueError) as e:
            print(f"Processing Error: {str(e)}")
            if raw_output:
                print(f"Problematic response snippet: {raw_output[:500]}...")
            return None
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return None

    def clean_json_output(self, raw_output):
        """Advanced JSON extraction with multiple fallback patterns"""
        # Try markdown code block first
        code_block_match = re.search(r'```json\s*({.*?})\s*```', raw_output, re.DOTALL)
        if code_block_match:
            return code_block_match.group(1).strip()

        # Fallback to plain JSON detection
        json_match = re.search(r'\{\s*"patient_info".*?"treatment_plan".*?\}', raw_output, re.DOTALL)
        if json_match:
            return json_match.group(0).strip()

        # Final fallback: Find first { and last }
        start_idx = raw_output.find('{')
        end_idx = raw_output.rfind('}') + 1
        if start_idx != -1 and end_idx != 0:
            return raw_output[start_idx:end_idx]

        return ""


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
    output_folder = "../Claude_case_study_json"
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

