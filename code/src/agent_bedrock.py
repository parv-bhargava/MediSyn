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
            response = self.client.invoke_model(
                modelId=model_id,
                body=json.dumps(body_content),
            )
            return response['body'].read()
        except ClientError as e:
            print(f"Error: {e}")
            return None


if __name__ == '__main__':
    client = LLM()
    model_id = "meta.llama3-70b-instruct-v1:0"
    body_content = {
        "input": "What is the capital of France?",
        "parameters": {
            "temperature": 0.7,
            "max_tokens": 100
        }
    }
    response = client.get_response(model_id, body_content)
    print(response)
