import asyncio
import logging
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv


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
            final_answer_prompt="Please provide a final treatment plan based on discussion."
        )

    async def trigger(self, case_study: str, discussion_message: str = None, stream: bool = True):
        """
        Trigger the agent system with the provided case study and discussion message.

        Args:
            case_study (str): The case study details.
            discussion_message (str, optional): An additional discussion prompt.
                Defaults to a preset message if not provided.
            stream (bool, optional): If True, yields messages as they are generated.
                If False, aggregates the response and yields a single final result.

        Yields:
            When stream is True: yields messages (or TaskResult) as they are produced.
            When stream is False: yields one aggregated string containing all agent responses.
        """
        if discussion_message is None:
            discussion_message = "Let's have a discussion on how we can provide the best care for this case."

        prompt = (
            f"IPE Case Study:\n{case_study}\n\n"
            f"Discussion: {discussion_message}\n\n"
            "Please provide your interprofessional perspectives on this case."
        )

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
    async def main():
        logging.basicConfig(
            filename="agent_discussion.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

        agent_system = InterprofessionalAgentSystem()

        case_study = (
            "Patient A is a 65-year-old male with a history of hypertension and type 2 diabetes, "
            "presenting with shortness of breath and chest pain. Examination revealed bilateral rales "
            "and an S3 heart sound, suggesting congestive heart failure. Further lab tests and imaging are pending."
        )

        # --- Streaming Mode ---
        print("Triggering agent system in streaming mode:\n")
        async for response in agent_system.trigger(case_study=case_study, stream=True):
            if isinstance(response, TaskResult):
                print("\nTask completed. Stop reason:", response.stop_reason)
            else:
                print(f"{response.source}:\n{response.content}\n")

        # # --- Aggregated Mode ---
        # print("\nTriggering agent system in aggregated (non-streaming) mode:\n")
        # aggregated_messages = ""
        # async for response in agent_system.trigger(case_study=case_study, stream=False):
        #     aggregated_messages = response
        # print(aggregated_messages)


    asyncio.run(main())
