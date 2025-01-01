from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task

# If you want to run a snippet of code before or after the crew starts, 
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

from crewai_tools import JSONSearchTool

json_search_tool = JSONSearchTool(

	json_path = '/home/sachin/Music/NFL/SportsRadar_API/nfl_injuries/2023/nfl_injuries_2023_PRE_week1.json',

    config={
        "llm": {
            "provider": "ollama",  # Other options include google, openai, anthropic, llama2, etc.
            "config": {
                "model": "gemma2",
                # Additional optional configurations can be specified here.
                # temperature=0.5,
                # top_p=1,
                # stream=true,
            },
        },
        "embedder": {
            "provider": "ollama", # or openai, ollama, ...
            "config": {
                "model": "mxbai-embed-large",
                "task_type": "retrieval_document",
                # Further customization options can be added here.
            },
        },
    }
)

@CrewBase
class NflPredicitonAssistant():
	"""NflPredicitonAssistant crew"""

	# Learn more about YAML configuration files here:
	# Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
	# Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	# If you would like to add tools to your agents, you can learn more about it here:
	# https://docs.crewai.com/concepts/agents#agent-tools
	@agent
	def researcher(self) -> Agent:
		return Agent(
			config=self.agents_config['researcher'],
			verbose=True,
			llm = LLM(model = 'ollama/gemma2', base_url = 'http://localhost:11434'),
			tools=[json_search_tool],
		)

	@agent
	def reporting_analyst(self) -> Agent:
		return Agent(
			config=self.agents_config['reporting_analyst'],
			verbose=True,
			llm = LLM(model = 'ollama/gemma2', base_url = 'http://localhost:11434'),
		)

	# To learn more about structured task outputs, 
	# task dependencies, and task callbacks, check out the documentation:
	# https://docs.crewai.com/concepts/tasks#overview-of-a-task
	@task
	def research_task(self) -> Task:
		return Task(
			config=self.tasks_config['research_task'],
		)

	@task
	def reporting_task(self) -> Task:
		return Task(
			config=self.tasks_config['reporting_task'],
			output_file='report.md'
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the NflPredicitonAssistant crew"""
		# To learn how to add knowledge sources to your crew, check out the documentation:
		# https://docs.crewai.com/concepts/knowledge#what-is-knowledge

		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)
