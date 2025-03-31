import os
import json
import re
import glob
from json import JSONDecodeError
from unstructured.partition.pdf import partition_pdf
import boto3
from dotenv import load_dotenv
from unstructured.documents.elements import Element

from code.src.agents.base_agent import Agent
from pydantic import BaseModel
from typing import List

# Load environment variables
load_dotenv()
PDF_PROCESSOR = """
You are a medical document processor. Extract ALL information into this EXACT format:
     "name": "string", 
     "age": "string",
     "diagnosis": ["string"]
     "summary": "Concise clinical summary",
     "team_members": ["string"],
     "background": "Medical history and key events",
     "assessment_plan": "The assessment plan section including the individual team member assessments",
     "assessment_results": "The assessment results section including the individual team member assessments results",
     "treatment_plan": "The Treatment plan section including the individual team member's take on the treatment plan"

    RULES:
    1. Output ONLY Exact Format given 
    2. Use exact field names
    3. Never add comments
    4. Maintain original medical terms
    5. Be explanatory and don't summarize the information
"""


class MedicalDocument(BaseModel):
    name: str
    age: str
    diagnosis: List[str]
    summary: str
    team_members: List[str]
    background: str
    assessment_plan: str
    assessment_results: str
    treatment_plan: str

    def to_nested_json(self) -> dict:
        return {
            "patient_info": {
                "name": self.name,
                "age": self.age,
                "diagnosis": self.diagnosis
            },
            "summary": self.summary,
            "team_members": self.team_members,
            "background": self.background,
            "assessment_plan": self.assessment_plan,
            "assessment_results": self.assessment_results,
            "treatment_plan": self.treatment_plan
        }


class PDFProcessor:
    def extract_pdf_text(self, pdf_path):
        """Extract structured text using Unstructured library"""
        elements = partition_pdf(
            filename=pdf_path,
            strategy="fast",
            infer_table_structure=True,
            include_page_breaks=False,
        )
        return "\n\n".join([str(el) for el in elements])

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
            response = Agent(name="PDF FORMAT", role=PDF_PROCESSOR, input=f"PDF CONTENT:\n{raw_text}",
                             model_id="gpt-40").structure_run(MedicalDocument)

            return response.to_nested_json()

        except (JSONDecodeError, ValueError) as e:
            print(f"Processing Error: {str(e)}")
            if raw_output:
                print(f"Problematic response:\n{raw_output}")
            return None
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return None

    def save_to_file(self, data, output_path):
        """Save processed data to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved results to {output_path}")


# Usage example
if __name__ == "__main__":
    processor = PDFProcessor()

    # Set up paths
    input_folder = os.getcwd()
    output_folder = "../gpt_case_study_json"
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
