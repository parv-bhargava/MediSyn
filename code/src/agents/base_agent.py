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
    def __init__(self, name, role, conversation="Conversation", model_id=None):
        """
        Superclass representing a generic agent.

        :param name: The identifier of the agent.
        :param role: The system prompt or primary instruction.
        :param conversation: Additional conversation or user prompt.
        :param model_id: The model identifier to determine which runner to use.
        """
        self.name = name
        self.role = role
        self.conversation = conversation
        self.model_id = model_id

    def run(self):
        """
        Execute the agent's task by delegating to the appropriate runner.
        Passes the agent's id and model_id to the runner.
        """
        if self.model_id in ["gpt-4o", "gpt-4o-turbo"]:
            return RunnerGPT(
                agent_id=self.name,
                role=self.role,
                conversation=self.conversation,
                model_id=self.model_id
            ).run()
        elif self.model_id in ["meta.llama3-70b-instruct-v1:0", "anthropic.claude-3-5-sonnet-20240620-v1:0"]:
            return RunnerBedrock(
                agent_id=self.name,
                role=self.role,
                conversation=self.conversation,
                model_id=self.model_id
            ).run()
        else:
            raise ValueError("Unsupported model ID.")


class RunnerGPT(Agent):
    def __init__(self, agent_id, role, conversation, model_id):
        """
        Initialize a RunnerGPT instance with OpenAI's GPT settings.

        :param agent_id: The identifier of the agent.
        :param role: The system prompt.
        :param conversation: The conversation text.
        :param model_id: The model identifier.
        """
        super().__init__(agent_id, role, conversation, model_id)

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


class RunnerBedrock(Agent):
    def __init__(self, agent_id, role, conversation="No Conversation", model_id=None, region_name="us-east-1"):
        """
        Initialize a RunnerBedrock instance with AWS Bedrock settings.

        :param agent_id: The identifier of the agent.
        :param role: The system prompt.
        :param conversation: The conversation text.
        :param model_id: The Bedrock model identifier.
        :param region_name: AWS region name.
        """
        super().__init__(agent_id, role, conversation, model_id)
        self.region_name = region_name
        self.client = None
        self.model_id = model_id  # set the model id from parameter
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

    def set_max_tokens(self, max_tokens=2064):
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

        if "anthropic.claude" in self.model_id:
            self.set_max_tokens(5000)
            body_content = {
                "anthropic_version": "bedrock-2023-05-31",
                "system": self.role,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": self.conversation
                            }
                        ]
                    }
                ],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            }
            try:
                response = self.client.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(body_content)
                )
                response_body = json.loads(response['body'].read().decode())
                print(f"Response from Bedrock (Claude) for agent {self.name}: {response_body}")
                return response_body['content'][0]['text'].strip()
            except ClientError as e:
                print(f"Error: {e.response['Error']['Message']}")
                return None
            except (KeyError, json.JSONDecodeError):
                print("Unexpected response format or failed to decode response for Claude model")
                return None

        elif "llama" in self.model_id.lower():
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
                print(f"Response from Bedrock (LLama) for agent {self.name}: {response_body}")
                return response_body['generation'].strip()
            except ClientError as e:
                print(f"Error: {e.response['Error']['Message']}")
                return None
            except (KeyError, json.JSONDecodeError):
                print("Unexpected response format or failed to decode response")
                return None
        else:
            raise ValueError("Unsupported model type. Please use a Claude or Llama model.")


# Example usage:
if __name__ == "__main__":
    try:
        # Example with a Claude model:
        agent_claude = Agent(
            name="AgentClaude",
            role="You are a medical document processor.",
            conversation="Extract the key details from the provided document.",
            model_id="anthropic.claude-3-5-sonnet-20240620-v1:0"
        )

        print("AgentClaude response (Claude):", agent_claude.run())

        agent_gpt = Agent(
            name="AgentGPT",
            role="You are a creative storyteller.",
            conversation="Generate a short story about space exploration.",
            model_id="gpt-4o"
        )
        print("AgentGPT response:", agent_gpt.run())

        agent_llama = Agent(
            name="AgentLLama",
            role="You are a medical document processor.",
            conversation="Extract the key details from the provided document.",
            model_id="meta.llama3-70b-instruct-v1:0"
        )
    except ValueError as e:
        print("Initialization error:", e)
