import boto3
import json
import os
from botocore.exceptions import ClientError
from dotenv import load_dotenv


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
                You are an expert nurse. 
                Provide your professional opinion concisely and explain your answers true to your knowledge.<|eot_id|>
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

if __name__ == '__main__':
    client = LLM()
    # model_id = "meta.llama3-70b-instruct-v1:0"
    # model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    model_id = "anthropic.claude-3-opus-20240229-v1:0"
    query = "What's the recommended treatment for migraine?"
    # query = "I have a fever. What should I do?"
    body_content = {
        "prompt": query,
        "temperature": 0.7,  # More creative but focused
        "max_gen_len": 1024,  # Allow longer responses
        "top_p": 0.95  # Broader token sampling
    }
    # response = client.get_response(model_id, body_content)
    # print(response)
    nurse_system_prompt = """You are an expert nurse.
    Provide your professional opinion concisely and explain your answers true to your knowledge."""
    response = client.get_claude_response(
        system_prompt=nurse_system_prompt,
        user_query=query,
        max_tokens=1024,
        temperature=0.2
    )
    print(response)