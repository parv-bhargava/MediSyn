import asyncio
import json
import logging
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_core import CancellationToken
from semantic_kernel import Kernel
from autogen_ext.models.semantic_kernel import SKChatCompletionAdapter
from semantic_kernel.connectors.ai.bedrock.bedrock_prompt_execution_settings import BedrockChatPromptExecutionSettings
from semantic_kernel.connectors.ai.bedrock.services.bedrock_chat_completion import BedrockChatCompletion
from semantic_kernel.memory.null_memory import NullMemory
from dotenv import load_dotenv

current_dir = os.path.dirname(__file__)
json_path = os.path.join(current_dir, '..', 'data', 'case-study-25.json')
CASE_STUDY_PATH = os.path.abspath(json_path)

#TODO: Have to fix bedrock chat completion

class InterprofessionalAgentSystem:
    def __init__(self, ai_model_id="meta.llama3-70b-instruct-v1:0", temperature=0.5):
        """
        Initialize the interprofessional agent system using the Semantic Kernel client.

        Args:
            host (str): The host URL for the Ollama server.
            ai_model_id (str): The AI model identifier.
            temperature (float): Temperature setting for the model.
        """
        load_dotenv()

        # Create the Ollama client.
        sk_client = BedrockChatCompletion(
            model_id=ai_model_id,
            env_file_path=".env",
        )
        bedrock_settings = BedrockChatPromptExecutionSettings(
            temperature=temperature,
        )
        # Wrap the Ollama client with the Semantic Kernel adapter.
        model_client = SKChatCompletionAdapter(
            sk_client, kernel=Kernel(memory=NullMemory()), prompt_settings=bedrock_settings
        )
        self.model_client = model_client
        self.model_client_orc = model_client

        self.nurse_agent = AssistantAgent(
            name="Nurse",
            model_client=self.model_client,
            system_message=(
                "user:You are a caring nurse. Provide insights focusing on patient care, "
                "empathy, and practical nursing observations in your response. Answer in not more than 90 words."
            )
        )

        self.physician_agent = AssistantAgent(
            name="Physician",
            model_client=self.model_client,
            system_message=(
                "user:You are a knowledgeable physician. Provide insights focusing on medical diagnostics, "
                "clinical decision-making, and treatment planning. Answer in not more than 90 words."
            )
        )

        self.social_worker_agent = AssistantAgent(
            name="SocialWorker",
            model_client=self.model_client,
            system_message=(
                "user:You are a compassionate social worker. Provide insights focusing on social care, "
                "support services, and community resources. Answer in not more than 90 words."
            )
        )

        termination_condition = MaxMessageTermination(7)
        self.team = MagenticOneGroupChat(
            [self.nurse_agent, self.physician_agent, self.social_worker_agent],
            model_client=self.model_client_orc,
            termination_condition=termination_condition,
            final_answer_prompt="""            
            **Required Output Format**
            Create a comprehensive treatment plan with specific recommendations from each team member's perspective.

            Structure your response using EXACTLY these section headers:
            {{For each team member}} 
            [Full Team Member Role Name]: [Recommendations]"""

        )

    async def trigger(self, prompt: str, discussion_message: str = None, stream: bool = True):
        cancellation_token = CancellationToken()
        stream_obj = self.team.run_stream(task=prompt, cancellation_token=cancellation_token)

        if stream:
            async for msg in stream_obj:
                yield msg
        else:
            aggregated_response = ""
            async for msg in stream_obj:
                if not isinstance(msg, TaskResult):
                    aggregated_response += f"{msg.source}:\n{msg.content}\n\n"
            yield aggregated_response


if __name__ == "__main__":
    def generate_prompt(case_study_path):
        with open(case_study_path, 'r', encoding='utf-8') as f:
            case_data = json.load(f)
        # Construct detailed prompt
        team_list = '\n'.join([f"- {role}" for role in case_data['team_members']])
        prompt = f"""
        user:
            **IPE Case Study Analysis Task**

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
            """
        return prompt

    def save_output(case_study_path: str, generated_text: str):
        """Save generated output with case-specific filename"""
        output_dir = "../data/outputs"
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


    async def main():
        logging.basicConfig(
            filename="agent_discussion.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

        agent_system = InterprofessionalAgentSystem()
        prompt = generate_prompt(CASE_STUDY_PATH)

        # --- Streaming Mode ---
        print("Triggering agent system in streaming mode:\n")
        async for response in agent_system.trigger(prompt=prompt, stream=True):
            if isinstance(response, TaskResult):
                print("\nTask completed. Stop reason:", response.stop_reason)
            else:
                print(f"{response.source}:\n{response.content}\n")

        # # --- Aggregated Mode ---
        # print("\nTriggering agent system in aggregated (non-streaming) mode:\n")
        # aggregated_messages = ""
        # async for response in agent_system.trigger(prompt=prompt, stream=False):
        #     aggregated_messages = response
        #     save_output(CASE_STUDY_PATH, aggregated_messages)
        # print(aggregated_messages)


    asyncio.run(main())

