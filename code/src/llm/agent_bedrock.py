import boto3
import json
import os
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class LLM:
    """
    LLM class to interact with Bedrock Models.
    """

    def __init__(self, region_name="us-east-1"):
        """
        Initialize the BedrockClient instance.

        :param region_name: AWS region for the Bedrock client, default is 'us-east-1'.
        """
        self.region_name = region_name
        self.aws_access_key_id = None
        self.aws_secret_access_key = None
        self.aws_session_token = None
        self.client = None
        self._load_credentials()

    def _load_credentials(self):
        """
        Load AWS credentials from the environment variables using dotenv.
        """
        load_dotenv()
        self.aws_access_key_id = os.getenv("aws_access_key_id")
        self.aws_secret_access_key = os.getenv("aws_secret_access_key")
        self.aws_session_token = os.getenv("aws_session_token")

    def create_client(self):
        """
        Create and return a Bedrock client.

        :return: boto3 client for Bedrock Runtime.
        """
        if not all([self.aws_access_key_id, self.aws_secret_access_key]):
            raise ValueError("AWS credentials are not properly set in the environment.")

        session = boto3.Session(
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token
        )
        self.client = session.client("bedrock-runtime", region_name=self.region_name)
        return self.client

    def get_response(self, model_id, body_content):
        """
        Invoke a model and return its response.

        :param model_id: The ID of the model to invoke.
        :param body_content: The request body content as a Python dictionary.
        :return: The model response or None in case of an error.
        """
        if self.client is None:
            self.create_client()

        try:
            # Llama formatted instructions
            if "llama3" in model_id:
                body_content["prompt"] = f"""<|begin_of_text|>
                        <|start_header_id|>system<|end_header_id|>
                        You are a helpful assistant. Provide accurate and detailed responses.<|eot_id|>
                        <|start_header_id|>user<|end_header_id|>
                        {body_content["prompt"]}<|eot_id|>
                        <|start_header_id|>assistant<|end_header_id|>"""

            response = self.client.invoke_model(
                modelId=model_id,
                body=json.dumps(body_content),
            )
            # Parse response properly
            response_body = json.loads(response['body'].read())
            return response_body['generation'].strip()

        except ClientError as e:
            print(f"Error: {e.response['Error']['Message']}")
            return None
        except KeyError:
            print("Unexpected response format from model")
            return None
        except json.JSONDecodeError:
            print("Failed to decode model response")
            return None

    def get_claude_response(self, system_prompt, model_id, user_query, max_tokens=500, temperature=0.3):
        """
        Generate agent responses using Claude models.

        :param system_prompt (str): Detailed instructions defining the AI's role, response format requirements.
        :param model_id (str): The ID of the model to invoke
        :param user_query (str): The patient case description or medical question to analyze
        :param max_tokens (int, optional): Maximum length of response in tokens (range: 100-4096).
                                       Clinical responses typically require 300-1000 tokens.
                                       Default: 500
        :param temperature (float, optional): Controls response creativity (range: 0.0-1.0).

        Returns:
            str | None: Generated medical recommendation as plain text with:
                       Returns None if error occurs
        """

        if self.client is None:
            self.create_client()

        try:
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": user_query
                            }
                        ]
                    }
                ],
                "system": system_prompt,
                "max_tokens": max_tokens,
                "temperature": max(0.0, min(1.0, temperature)),
                "top_p": 0.9,
                "stop_sequences": ["\n\nPatient", "</response>"]
            }

            response = self.client.invoke_model(
                modelId=model_id,
                body=json.dumps(body)
            )

            response_body = json.loads(response['body'].read())
            return response_body['content'][0]['text'].strip()

        except ClientError as e:
            print(f"AWS Error: {e.response['Error']['Message']}")
            return None
        except KeyError:
            print("Unexpected response format")
            return None

    def generate_treatment_plan(self, case_study_path):
        with open(case_study_path, 'r', encoding='utf-8') as f:
            case_data = json.load(f)
        # Construct detailed prompt
        team_list = '\n'.join([f"- {role}" for role in case_data['team_members']])
        prompt = f"""**IPE Case Study Analysis Task**

            **Patient Information**
            Name: {case_data['patient_info']['name']}
            Age: {case_data['patient_info']['age']}
            Diagnoses: {', '.join(case_data['patient_info']['diagnosis'])}

            **Case Summary**
            {case_data['summary']}

            **Medical Background**
            {case_data['background']}

            **Assessment Plan**
            {case_data['assessment_plan']}

            **Assessment Results**
            {case_data['assessment_results']}

            **Team Members**
            {team_list}

            **Required Output Format**
            Create a comprehensive treatment plan with specific recommendations from each team member's perspective.
            
            Structure your response using EXACTLY these section headers:
            {{For each team member}} 
            [Full Team Member Role Name]: [Recommendations]
            """

        # Generate response using LLaMA model
        body_content = {
            "prompt": prompt,
            "temperature": 0.3,  # Lower for factual accuracy
            "max_gen_len": 1500,
            "top_p": 0.9
        }

        response = self.get_response(
            model_id="meta.llama3-70b-instruct-v1:0",
            body_content=body_content
        )

        return response

    def save_output(self, case_study_path: str, generated_text: str):
        """Save generated output with case-specific filename"""
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)

        # Extract base filename and create output name
        base_name = os.path.splitext(os.path.basename(case_study_path))[0]
        output_filename = f"treatment-plan-{base_name}.json"
        output_path = os.path.join(output_dir, output_filename)

        output_data = {
            "source_file": case_study_path,
            "generated_text": generated_text
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        return output_path


if __name__ == '__main__':
    client = LLM()
    os.chdir("../../../Case_Study_Data")

    model_id = "meta.llama3-70b-instruct-v1:0"
    case_study_path = "case-study-25.json"
    result = client.generate_treatment_plan(case_study_path)
    client.save_output(case_study_path, result)
    print("IPE Treatment Plan Generated and Saved!!")