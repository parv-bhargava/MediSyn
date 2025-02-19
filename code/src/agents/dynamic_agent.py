#Dynamic agent manager that creates and deploys agents based on roles and prompts.
import os
import time
import uuid
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
# from langchain_aws import BedrockLLM
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

### OpenAI GPT-4o Agent Implementation
class Agent:
    def __init__(self, name, prompt):
        self.name = name
        self.prompt = prompt
        self.id = uuid.uuid4()

    def run(self):
        time.sleep(1)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": self.prompt},
            ]).choices[0].message.content

        if not response:
            response = "No response."
        print(f'response: {response}')
        return response


class DynamicAgentManager:
    def __init__(self):
        self.agents = {}
        self.memory = {}
        self.event_queue = []

    def _agent_prompt_generator(self, role):
        prompt_text = f"Generate a prompt for this role as a {role}."
        prompt_generator = Agent(name="PROMPT_GENERATOR", prompt=prompt_text)
        generated_prompt = prompt_generator.run()
        print(generated_prompt)
        return generated_prompt

    def _create_and_deploy_agent(self, role, base_prompt):
        prompt = self._agent_prompt_generator(role)
        agent_name = f"Agent_{role}"
        agent = Agent(name=agent_name, prompt=prompt)
        self.agents[agent.id] = agent
        self.event_queue.append(('deploy', agent))
        print(f"[{datetime.now()}] {agent.name} created and scheduled for deployment.")
        return agent.id

    def _execute_agents(self):
        while self.event_queue:
            event_type, agent = self.event_queue.pop(0)
            if event_type == 'deploy':
                response = agent.run()
                self.memory[str(agent.id)] = response
                print(f"[{datetime.now()}] {agent.name} executed. Response stored.")

    def run_manager(self, roles, base_prompt):
        for role in roles:
            self._create_and_deploy_agent(role, base_prompt)
        self._execute_agents()


def main():
    roles = ["physician", "oncologist", "nurse"]
    base_prompt = "Prepare a treatment plan for a patient."

    manager = DynamicAgentManager()
    manager.run_manager(roles, base_prompt)

    print("\nStored Agent Responses:")
    for agent_id, response in manager.memory.items():
        print(f"Agent ID: {agent_id}\nResponse: {response}\n")


if __name__ == "__main__":
    main()
