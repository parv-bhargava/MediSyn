# Dynamic agent manager that creates and deploys agents based on roles and prompts.
from datetime import datetime

from base_agent import Agent
from configs import PROMPT_GENERATOR_PROMPT


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
    """

    def __init__(self):
        self.agents = {}
        self.memory = {}
        self.event_queue = []

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
        prompt_text = PROMPT_GENERATOR_PROMPT.format(role)
        prompt_generator = Agent(name="PROMPT_GENERATOR", prompt=prompt_text)
        generated_prompt = prompt_generator.run()
        return generated_prompt

    def _create_and_deploy_agent(self, role: object, base_prompt: object) -> str:
        """
        Creates and deploys an agent based on the provided role and base prompt. The method
        generates a prompt for the agent, creates an instance of the `Agent` class with the
        specified properties, stores it in the agents dictionary, and schedules it for
        deployment by appending to the event queue.

        :param role: The role assigned to the agent. Determines the functional behavior of
            the agent.
        :type role: str
        :param base_prompt: A base prompt to initialize the agent's operations or context.
            Sets the initial guidance for the agent.
        :type base_prompt: str
        :return: The name of the created agent, uniquely generated combining the role and
            an identifier.
        :rtype: str
        """
        prompt = self._agent_prompt_generator(role)
        agent_name = f"Agent_{role}"
        agent = Agent(name=agent_name, prompt=prompt, base_prompt=base_prompt)
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
        synthesis_base_prompt = (
            f"Based on the following agent responses, synthesize a final treatment plan:\n\n"
            f"{combined_responses}"
        )
        synthesis_system_prompt = self._agent_prompt_generator("synthesis")
        synthesis_agent = Agent(name="SYNTHESIS_AGENT", prompt=synthesis_system_prompt,
                                base_prompt=synthesis_base_prompt)
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

    def run_manager(self, roles: object, base_prompt: object, synthesis: object = False) -> None:
        """
        Executes a sequence of operations by creating and deploying agents based on the
        provided roles and base prompt. Optionally creates and deploys a synthesis agent
        if the synthesis parameter is set to True. The function coordinates the execution
        of all deployed agents.

        :param roles: A list of roles representing the types of agents to be created
            and deployed.
        :type roles: list
        :param base_prompt: A shared prompt to initialize all created agents.
        :type base_prompt: str
        :param synthesis: Indicates whether a synthesis agent should be created and
            deployed in addition to the standard agents.
        :type synthesis: bool
        :return: None
        """
        for role in roles:
            self._create_and_deploy_agent(role, base_prompt)
        self._execute_agents()
        if synthesis:
            self._create_and_deploy_synthesis_agent()
            self._execute_agents()
