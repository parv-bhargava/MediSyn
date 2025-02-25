import asyncio
import json
import logging
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

current_dir = os.path.dirname(__file__)
json_path = os.path.join(current_dir, '..', 'data', 'case-study-25.json')
CASE_STUDY_PATH = os.path.abspath(json_path)


class InterprofessionalAgentSystem:
    def __init__(self, openai_api_key: str = None):
        """
        Initialize the interprofessional agent system.

        Args:
            openai_api_key (str, optional): Your OpenAI API key. If not provided,
                                            it will be read from the environment variable OPENAI_API_KEY.
        """
        load_dotenv()
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError(
                "OpenAI API Key must be provided either as a parameter or in the environment variable OPENAI_API_KEY")

        self.model_client = OpenAIChatCompletionClient(model="gpt-4o", api_key=self.openai_api_key)
        self.model_client_orc = OpenAIChatCompletionClient(model="o3-mini-2025-01-31", api_key=self.openai_api_key)

        self.nurse_agent = AssistantAgent(
            name="Nurse",
            model_client=self.model_client,
            system_message=(
                "You are a caring nurse. Provide insights focusing on patient care, "
                "empathy, and practical nursing observations in your response. Answer in not more than 90 words."
            )
        )

        self.physician_agent = AssistantAgent(
            name="Physician",
            model_client=self.model_client,
            system_message=(
                "You are a knowledgeable physician. Provide insights focusing on medical diagnostics, "
                "clinical decision-making, and treatment planning. Answer in not more than 90 words."
            )
        )

        self.social_worker_agent = AssistantAgent(
            name="SocialWorker",
            model_client=self.model_client,
            system_message=(
                "You are a compassionate social worker. Provide insights focusing on social care, "
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
            """
        return prompt

    def save_output(case_study_path: str, generated_text: str):
        """Save generated output with case-specific filename"""
        output_dir = "../../data/outputs"
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
        # print("Triggering agent system in streaming mode:\n")
        # async for response in agent_system.trigger(prompt=prompt, stream=True):
        #     if isinstance(response, TaskResult):
        #         print("\nTask completed. Stop reason:", response.stop_reason)
        #     else:
        #         print(f"{response.source}:\n{response.content}\n")

        # # --- Aggregated Mode ---
        print("\nTriggering agent system in aggregated (non-streaming) mode:\n")
        aggregated_messages = ""
        async for response in agent_system.trigger(prompt=prompt, stream=False):
            aggregated_messages = response
            save_output(CASE_STUDY_PATH, aggregated_messages)
        print(aggregated_messages)


    asyncio.run(main())
