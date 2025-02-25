from openai import OpenAI
from dotenv import load_dotenv
# from langchain_aws import BedrockLLM
import os
import time
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Agent:
    def __init__(self, name, prompt, base_prompt=""):
        """
        Represents an entity or object with specific attributes: name, prompt, and an
        optional base_prompt.

        This class is a basic structure designed to encapsulate the properties provided
        during initialization, enabling operations or further extensions concerning
        `name`, `prompt`, and `base_prompt`.

        :param name: The identifier of the entity, acting as a name or title.
        :type name: str
        :param prompt: A string prompt representing primary data or textual context
            associated with the entity.
        :type prompt: str
        :param base_prompt: An optional foundational prompt or default textual context
            for the entity. Defaults to an empty string.
        :type base_prompt: str
        """
        self.name = name
        self.prompt = prompt
        self.base_prompt = base_prompt

    def run(self):
        time.sleep(1)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": self.base_prompt},
            ]).choices[0].message.content

        if not response:
            response = "No response."
        return response

### Bedrock LLM Agent Implementation
#
# class Agent:
#     def __init__(self, name, prompt):
#         self.name = name
#         self.prompt = prompt
#         self.id = uuid.uuid4()
#
#     def run(self):
#         time.sleep(1)
#         llm = BedrockLLM(
#             credentials_profile_name="bedrock-client",
#             region_name="us-east-1",
#             model="meta.llama3-70b-instruct-v1:0"
#         )
#
#         response = llm.invoke(self.prompt)
#         if response == "":
#             response = "No response."
#         print(f'response:{response}')
#         return response