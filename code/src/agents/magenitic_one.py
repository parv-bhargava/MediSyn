import asyncio
import logging
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.base import TaskResult
from dotenv import load_dotenv

logging.basicConfig(
    filename="agent_discussion.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

async def main():
    model_client = OpenAIChatCompletionClient(model="gpt-4o", api_key=OPENAI_API_KEY)
    model_client_orc = OpenAIChatCompletionClient(model="o3-mini-2025-01-31", api_key=OPENAI_API_KEY)
    nurse_agent = AssistantAgent(
        name="Nurse",
        model_client=model_client,
        system_message=(
            "You are a caring nurse. Provide insights focusing on patient care, "
            "empathy, and practical nursing observations in your response. Answer in not more than 90 words."
        )
    )

    physician_agent = AssistantAgent(
        name="Physician",
        model_client=model_client,
        system_message=(
            "You are a knowledgeable physician. Provide insights focusing on medical diagnostics, "
            "clinical decision-making, and treatment planning. Answer in not more than 90 words"
        )
    )

    social_worker_agent = AssistantAgent(
        name="SocialWorker",
        model_client=model_client,
        system_message=(
            "You are a compassionate social worker. Provide insights focusing on social care, "
            "support services, and community resources. Answer in not more than 90 words."
        )
    )

    termination_condition = MaxMessageTermination(7)
    team = MagenticOneGroupChat(
        [nurse_agent, physician_agent, social_worker_agent],
        model_client=model_client_orc,
        termination_condition=termination_condition,
        final_answer_prompt="Please provide a final treatment plan based on discussion."
    )

    case_study = (
        "Patient A is a 65-year-old male with a history of hypertension and type 2 diabetes, "
        "presenting with shortness of breath and chest pain. Examination revealed bilateral rales "
        "and an S3 heart sound, suggesting congestive heart failure. Further lab tests and imaging are pending."
    )
    discussion_message = (
        "Let's have discussion on how we can provide the best care for Patient A. "
    )

    prompt = (
        f"IPE Case Study:\n{case_study}\n\n"
        f"Discussion: {discussion_message}\n\n"
        "Please provide your interprofessional perspectives on this case."
    )

    stream = team.run_stream(task=prompt, cancellation_token=CancellationToken())

    print("\nStreaming Team Response:")
    logging.info("Streaming Team Response:")

    async for msg in stream:
        if isinstance(msg, TaskResult):
            print("\n\nTask completed. Stop reason:\n\n", msg.stop_reason)
        else:
            print(f"\n{msg.source}:\n\n{msg.content}")

if __name__ == "__main__":
    asyncio.run(main())
