# Dynamic agent manager that creates and deploys agents based on roles and prompts.
from datetime import datetime

from agents.base_agent import Agent
from configs.configs import PROMPT_GENERATOR_PROMPT ,SYNTHESIS_AGENT_PROMPT


class DynamicAgentManager:
    """
    DynamicAgentManager is a management class for creating, deploying, executing, and
    managing multiple agents within a dynamic, role-based AI system. This class allows
    for efficient orchestrating of agents which take on different roles, generate role-specific
    prompts, execute tasks, and optionally synthesize results into a final cohesive response.

    DynamicAgentManager aims to streamline the process of multi-agent handling by automating
    the creation, deployment, execution, and memory storage of responses for developers working
    with AI-driven systems.

    :ivar agents: A dictionary mapping agent names to their instances.
                  The agents are dynamically created and managed.
    :type agents: dict

    :ivar memory: A dictionary storing responses executed by agents.
                  This serves as a shared memory space for responses.
    :type memory: dict

    :ivar event_queue: A list-based event queue for scheduling actions such as
                       deployment of agents. It ensures correct processing order.
    :type event_queue: list

    :ivar model_id: The model ID used for agent creation and execution.
    :type model_id: str
    """

    def __init__(self, model_id=None):
        self.agents = {}
        self.memory = {}
        self.event_queue = []
        self.model_id = model_id

    def _agent_prompt_generator(self, role: object) -> str:
        """
        Generates a dynamic prompt based on the provided role. The method uses
        an instance of the Agent class to construct and execute the prompt
        generation logic. The output is tailored specifically to the supplied
        role, ensuring that the generated prompt aligns with the role's context.

        :param role: The role for which the prompt should be generated.
        :type role: str
        :return: The generated prompt based on the given role.
        :rtype: str
        """
        prompt_generator = Agent(name="PROMPT_GENERATOR", role=PROMPT_GENERATOR_PROMPT.format(role),
                                 model_id=self.model_id)
        generated_prompt = prompt_generator.run()
        return generated_prompt

    def _create_and_deploy_agent(self, role: str, input: str) -> str:
        """
        Creates and deploys an agent based on the given role and input parameters. This function
        generates a specific prompt for the agent's role, constructs the agent, adds it to the internal
        agents dictionary, schedules its deployment by appending it to the event queue, and logs creation
        information. Finally, it returns the name of the created agent.

        :param role: The role assigned to the agent to determine its behavior.
        :type role: str
        :param input: Initial input or context associated with the agent.
        :type input: str
        :return: The name of the created and deployed agent.
        :rtype: str
        """
        role_prompt = self._agent_prompt_generator(role)
        agent_name = f"Agent_{role}"
        agent = Agent(name=agent_name, role=role_prompt, input=input, model_id=self.model_id)
        self.agents[agent.name] = agent
        self.event_queue.append(('deploy', agent))
        print(f"[{datetime.now()}] {agent.name} created and scheduled for deployment.")
        return agent.name

    def _create_and_deploy_synthesis_agent(self) -> str:
        """
        Creates and deploys a synthesis agent by aggregating responses from other agents,
        generating a synthesis prompt, and scheduling the new agent for deployment.

        This method collects responses from all registered agents (excluding any whose
        names start with "Synthesis") and consolidates them into a formatted string.
        It then generates a base prompt to guide the synthesis agent's behavior and
        initializes the agent. Finally, the agent is stored, scheduled for deployment,
        and its name is returned.

        :return: 
        :param self: The current instance of the class.
        :return: The name of the newly created synthesis agent as a string.
        :rtype: str

        :raises: This method does not document exceptions explicitly.
        """
        combined_responses = "\n".join(
            [f"Response from {agent.name}: {self.memory.get(agent.name, 'No response')}"
             for name, agent in self.agents.items()
             if not agent.name.startswith("Synthesis")]
        )
        # # print(combined_responses, type(combined_responses))
        # combined_responses=''.join(combined_responses.split()[:200])
        synthesis_base_prompt = (
            f"Based on the following agent responses, synthesize a final treatment plan:\n\n"
            f"{combined_responses}"
        )
        synthesis_system_prompt = SYNTHESIS_AGENT_PROMPT
        synthesis_agent = Agent(name="SYNTHESIS_AGENT", role=synthesis_system_prompt,
                                input=synthesis_base_prompt, model_id=self.model_id)
        self.agents[synthesis_agent.name] = synthesis_agent
        self.event_queue.append(('deploy', synthesis_agent))
        print(f"[{datetime.now()}] {synthesis_agent.name} created and scheduled for deployment.")
        return synthesis_agent.name

    def _execute_agents(self) -> None:
        """
        Executes all agents present in the event queue and processes their actions based
        on the event type. This function iterates through the event queue, dispatches
        each agent to execute its respective actions, records their responses in memory,
        and logs the operation's completion status.

        :param self: An instance of the class encapsulating this function.

        :raises KeyError: If an invalid or unexpected event type is encountered in
            the event queue.
        """
        while self.event_queue:
            event_type, agent = self.event_queue.pop(0)
            if event_type == 'deploy':
                response = agent.run()
                self.memory[agent.name] = response
                print(f"[{datetime.now()}] {agent.name} executed. Response stored.")

    def run_manager(self, roles: [str], input: str, synthesis: object = False) -> None:
        """
        Executes and manages multiple agents based on the provided roles and input. Optionally,
        handles a synthesis process if specified. This function orchestrates the creation, deployment,
        and execution of agents.

        :param roles: A list of roles that determines the agents to be created and deployed.
        :type roles: list[str]
        :param input: The input content that agents use as context.
        :type input: str
        :param synthesis: Optional flag to indicate whether a synthesis agent should be created and executed.
        Defaults to False.
        :type synthesis: bool
        :return: None
        """
        for role in roles:
            self._create_and_deploy_agent(role, input)
        self._execute_agents()
        # If synthesis is enabled, create and deploy a synthesis agent
        if synthesis:
            self._create_and_deploy_synthesis_agent()
            self._execute_agents()


if __name__ == "__main__":
    manager = DynamicAgentManager(model_id="meta.llama3-70b-instruct-v1:0")
    roles = ["chemist", "engineer", "physicist"]
    base_prompt = "Develop a new technology based on the following specifications:"
    manager.run_manager(roles, base_prompt, synthesis=True)
    print("\nStored Agent Responses:")
    for agent_name, response in manager.memory.items():
        print(f"Agent ID: {agent_name}\nResponse: {response}\n")
        if agent_name == "SYNTHESIS_AGENT":
            print(f"Synthesized Response: {response}")