from dotenv import load_dotenv
import os
import time
import json
import boto3
from botocore.exceptions import ClientError

from openai import OpenAI
load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class Agent:
    def __init__(self, name, role, conversation=""):
        """
        Superclass representing a generic agent.

        :param name: The identifier of the agent.
        :param role: The system prompt or primary instruction.
        :param conversation: Additional conversation or user prompt.
        """
        self.name = name
        self.role = role
        self.conversation = conversation

    def run(self):
        """
        Execute the agent's task.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the run() method.")


class AgentGPT(Agent):
    def run(self):
        """
        Execute the agent's task using OpenAI's GPT API.
        """
        time.sleep(1)  # simulate delay
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": self.role},
                {"role": "user", "content": self.conversation},
            ]
        ).choices[0].message.content

        return response if response else "No response."


class AgentBedrock(Agent):
    def __init__(self, name, role, conversation="", region_name="us-east-1"):
        """
        Initialize an AgentBedrock instance with AWS Bedrock settings.

        :param name: The identifier of the agent.
        :param role: The system prompt or primary instruction.
        :param conversation: Additional conversation or user prompt.
        :param region_name: AWS region name.
        """
        super().__init__(name, role, conversation)
        self.region_name = region_name
        self.client = None
        self.model_id = None
        self.temperature = 0.7
        self.max_tokens = 1500
        self._load_credentials()

    def _load_credentials(self):
        """
        Load AWS credentials from environment variables and create a Bedrock client.
        """
        aws_access_key_id = os.getenv("aws_access_key_id")
        aws_secret_access_key = os.getenv("aws_secret_access_key")
        aws_session_token = os.getenv("aws_session_token")

        if not all([aws_access_key_id, aws_secret_access_key]):
            raise ValueError("AWS credentials are not properly set in the environment.")

        session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token
        )
        self.client = session.client("bedrock-runtime", region_name=self.region_name)

    def set_model(self, model_id):
        """
        Configure the model ID for inference.
        """
        self.model_id = model_id

    def set_temperature(self, temperature):
        """
        Set the response generation temperature.
        """
        self.temperature = max(0.0, min(1.0, temperature))

    def set_max_tokens(self, max_tokens):
        """
        Set the maximum tokens for response generation.
        """
        self.max_tokens = max_tokens

    def run(self):
        """
        Execute the agent's task using AWS Bedrock.
        Combines self.role and self.conversation before sending the request.
        """
        combined_prompt = f"{self.role}\n{self.conversation}"
        if not self.client:
            raise ValueError("AWS client not initialized.")
        if not self.model_id:
            raise ValueError("Model ID is not set.")

        body_content = {
            "prompt": combined_prompt,
            "temperature": self.temperature,
            "max_gen_len": self.max_tokens,
            "top_p": 0.9
        }
        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body_content)
            )
            response_body = json.loads(response['body'].read().decode())
            print(f"Response from Bedrock: {response_body}")
            return response_body['generation'].strip()
        except ClientError as e:
            print(f"Error: {e.response['Error']['Message']}")
            return None
        except (KeyError, json.JSONDecodeError):
            print("Unexpected response format or failed to decode response")
            return None



if __name__ == "__main__":
    try:
        agent_bedrock = AgentBedrock(
            name="AgentBedrock",
            role="You are a creative storyteller.",
            conversation="Generate a short story about space exploration."
        )
        agent_bedrock.set_model("meta.llama3-70b-instruct-v1:0")
        print("AgentBedrock response:", agent_bedrock.run())
    except ValueError as e:
        print("Initialization error:", e)
