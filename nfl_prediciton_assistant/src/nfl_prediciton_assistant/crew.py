from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from ._tools import json_to_embeddings, csv_to_embeddings
from .custom_embedder import EmbeddingTool




llm = LLM(model = 'ollama/gemma2', base_url = 'http://localhost:11434')


@CrewBase
class NflPredicitonAssistant():
	"""NflPredicitonAssistant crew"""

	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'
	# base_config = 'config/base_config.yaml'
	base_config = '/home/sachin/Music/NeevHQ/NFL_Prediction/nfl_prediciton_assistant/src/nfl_prediciton_assistant/config/base_config.yaml'


	@agent
	def data_embedding_agent(self) -> Agent:

		data_embedder = Agent(			
			config=self.agents_config['data_embedding_agent'],
			verbose=True,
			llm = llm,
			tools=[EmbeddingTool(config_path=self.base_config)],)
		
		return data_embedder

	@agent
	def consensus_agent(self) -> Agent:

		return Agent(
			config=self.agents_config['consensus_agent'],
			verbose=True,
			llm = llm,
		)

	
	@task
	def data_embedding_task(self) -> Task:

		return Task(
			config=self.tasks_config['data_embedding_task'],

		)

	@task
	def consensus_prediction_task(self) -> Task:
		return Task(
			config=self.tasks_config['consensus_prediction_task'],
			output_file='report.md'
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the NflPredicitonAssistant crew"""

		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
			memory=True,
			embedder={
				"provider": "ollama",
				"config": {"model": "mxbai-embed-large"}
    				},

			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)
