from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from ._tools import json_to_embeddings, csv_to_embeddings
from .custom_embedder import data_embedding_tool

# If you want to run a snippet of code before or after the crew starts, 
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators


llm = LLM(model = 'ollama/gemma2', base_url = 'http://localhost:11434')
@CrewBase
class NflPredicitonAssistant():
	"""NflPredicitonAssistant crew"""

	# Learn more about YAML configuration files here:
	# Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
	# Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'
	knowledge_config = 'config/knowledge.yaml'
	db_config = 'config/db.yaml'

	# If you would like to add tools to your agents, you can learn more about it here:
	# https://docs.crewai.com/concepts/agents#agent-tools
	@agent
	def data_preparation_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['data_preparation_agent'],
			verbose=True,
			llm = llm,
			tools=[data_embedding_tool()],
		)

	@agent
	def consensus_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['consensus_agent'],
			verbose=True,
			llm = llm,
		)

	# To learn more about structured task outputs, 
	# task dependencies, and task callbacks, check out the documentation:
	# https://docs.crewai.com/concepts/tasks#overview-of-a-task
	@task
	def data_preparation_task(self) -> Task:
		return Task(
			config=self.tasks_config['data_preparation_task'],

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
		# To learn how to add knowledge sources to your crew, check out the documentation:
		# https://docs.crewai.com/concepts/knowledge#what-is-knowledge

		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
			memory=True,
			embedder={
        	"provider": "ollama",
        	"config": {
            "model": "mxbai-embed-large"
        			}
    				},
			# long_term_memory=EnhanceLongTermMemory(
			# 	storage=LTMSQLiteStorage(
			# 		db_path="/my_data_dir/my_crew1/long_term_memory_storage.db"
			# 	)
			# ),
			# short_term_memory=EnhanceShortTermMemory(
			# 	storage=CustomRAGStorage(
			# 		crew_name="my_crew",
			# 		storage_type="short_term",
			# 		data_dir="//my_data_dir",
			# 		model=embedder["model"],
			# 		dimension=embedder["dimension"],
			# 	),
			# ),
			# entity_memory=EnhanceEntityMemory(
			# 	storage=CustomRAGStorage(
			# 		crew_name="my_crew",
			# 		storage_type="entities",
			# 		data_dir="//my_data_dir",
			# 		model=embedder["model"],
			# 		dimension=embedder["dimension"],
			# 	),
			# ),
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)
