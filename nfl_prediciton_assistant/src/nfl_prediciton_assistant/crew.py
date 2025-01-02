from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from .custom_embedder import EmbeddingTool
from .custom_retrieval import RetrievalTool
from .custom_websearch import WebSearch




llm = LLM(model = 'ollama/gemma2', base_url = 'http://localhost:11434')


@CrewBase
class NflPredicitonAssistant():
	"""NflPredicitonAssistant crew"""

	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'
	# base_config = 'config/base_config.yaml'
	base_config = '/home/sachin/Music/NeevHQ/NFL_Prediction/nfl_prediciton_assistant/src/nfl_prediciton_assistant/config/base_config.yaml'


	#sequence of agents
	@agent
	def data_embedding_agent(self) -> Agent:

		data_embedder = Agent(			
			config = self.agents_config['data_embedding_agent'],
			verbose = False,
			llm = llm,
			tools=[EmbeddingTool(config_path=self.base_config), WebSearch(config_path=self.base_config)],
		)
		print("Embedding Done")
		return data_embedder
	
	@agent
	def data_retrieval_agent(self) -> Agent:

		data_retrieval = Agent(
			config = self.agents_config['data_retrieval_agent'],
			verbose = True,
			llm = llm,
			tools = [RetrievalTool(config_path=self.base_config)],
		)
		print("Retrieval Done")
		return data_retrieval
	
	@agent
	def trend_analysis_agent(self) -> Agent:

		trend_analysis = Agent(
			config = self.agents_config['trend_analysis_agent'],
			verbose = True,
			llm = llm,
			tools = [RetrievalTool(config_path=self.base_config)],
		)
		print("Trend Analysis Done")
		return trend_analysis
	
	@agent
	def team_changes_agent(self) -> Agent:

		team_changes = Agent(
			config = self.agents_config['team_changes_agent'],
			verbose = True,
			llm = llm,
			tools = [RetrievalTool(config_path=self.base_config)],
		)
		print("Team change Analysis Done")
		return team_changes
	
	@agent
	def injury_analysis_agent(self) -> Agent:

		injury_analysis = Agent(
			config = self.agents_config['injury_analysis_agent'],
			verbose = True,
			llm = llm,
			tools = [RetrievalTool(config_path=self.base_config)],
		)
		print("Injury Analysis Done")
		return injury_analysis
	
	@agent
	def head_to_head_analysis_agent(self) -> Agent:

		head_to_head_analysis = Agent(
			config = self.agents_config['head_to_head_analysis_agent'],
			verbose = True,
			llm = llm,
			tools = [RetrievalTool(config_path=self.base_config)],
		)
		print("Team's head to head Analysis Done")
		return head_to_head_analysis

	@agent
	def current_season_performance_agent(self) -> Agent:

		current_season_performance = Agent(
			config = self.agents_config['current_season_performance_agent'],
			verbose = True,
			llm = llm,
			tools = [RetrievalTool(config_path=self.base_config)],
		)
		print("Current Season Performance Analysis Done")
		return current_season_performance	

	@agent
	def coaching_strategy_analysis_agent(self) -> Agent:

		coaching_strategy_analysis = Agent(
			config = self.agents_config['coaching_strategy_analysis_agent'],
			verbose = True,
			llm = llm,
			tools = [RetrievalTool(config_path=self.base_config)],
		)
		print("Coaching Stategy Analysis Done")
		return coaching_strategy_analysis
	
	@agent
	def environmental_impact_analysis_agent(self) -> Agent:

		environmental_impact_analysis = Agent(
			config = self.agents_config['environmental_impact_analysis_agent'],
			verbose = True,
			llm = llm,
			tools = [RetrievalTool(config_path=self.base_config)],
		)
		print("Impact of environment Analysis Done")
		return environmental_impact_analysis
	
	@agent
	def performance_summary_agent(self) -> Agent:

		performance_summary = Agent(
			config = self.agents_config['performance_summary_agent'],
			verbose = True,
			llm = llm,
		)
		print("Preformance Summary Done")
		return performance_summary

	@agent
	def consensus_agent(self) -> Agent:

		consensus_agent = Agent(
			config=self.agents_config['consensus_agent'],
			verbose=True,
			llm = llm,
		)
		print("Consensus Summary Done")
		return consensus_agent
	

	manager = Agent(
		role="Project Manager",
		goal="Efficiently manage the crew and ensure high-quality task completion",
		backstory="You're an experienced project manager, skilled in overseeing complex projects and guiding teams to success. Your role is to coordinate the efforts of the crew members, ensuring that each task is completed on time and to the highest standard.",
		allow_delegation=True,
		)
	

	#sequence of tasks
	@task
	def data_embedding_task(self) -> Task:
		
		embedding_task = Task(
			config=self.tasks_config['data_embedding_task'],
		)
		return embedding_task
	
	@task
	def data_retrieval_task(self) -> Task:

		retrieval_task = Task(
			config = self.tasks_config['data_retrieval_task'],
		)
		return retrieval_task

	@task
	def trend_analysis_task(self) -> Task:

		trend_analysis_task = Task(
			config = self.tasks_config['trend_analysis_task'],
		)
		return trend_analysis_task

	@task
	def team_changes_task(self) -> Task:

		team_changes_task = Task(
			config = self.tasks_config['team_changes_task'],
		)
		return team_changes_task


	@task
	def injury_analysis_task(self) -> Task:

		injury_analysis_task = Task(
			config = self.tasks_config['injury_analysis_task'],
		)
		return injury_analysis_task
	

	@task
	def head_to_head_analysis_task(self) -> Task:

		head_to_head_analysis_task = Task(
			config = self.tasks_config['head_to_head_analysis_task'],
		)
		return head_to_head_analysis_task
	
	@task
	def current_season_performance_task(self) -> Task:

		current_season_performance_task = Task(
			config = self.tasks_config['current_season_performance_task'],
		)
		return current_season_performance_task
	
	@task
	def coaching_strategy_task(self) -> Task:

		coaching_strategy_task = Task(
			config = self.tasks_config['coaching_strategy_task'],
		)
		return coaching_strategy_task
	
	@task
	def environmental_impact_task(self) -> Task:

		environmental_impact_task = Task(
			config = self.tasks_config['environmental_impact_task'],
		)
		return environmental_impact_task

	@task
	def performance_summary_task(self) -> Task:

		performance_summary_task = Task(
			config = self.tasks_config['performance_summary_task'],
			context = [self.environmental_impact_task, self.coaching_strategy_task]	
		)
		return performance_summary_task	
	@task
	def consensus_summary_task(self) -> Task:

		consensus_summary = Task(
			config=self.tasks_config['consensus_summary_task'],
			context = [self.performance_summary_task]
		)

		return consensus_summary
	

	@crew
	def crew(self) -> Crew:
		"""Creates the NflPredicitonAssistant crew"""

		crew_workflow = Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.hierarchical,
			verbose=True,
			memory=True,
			manager_agent=self.manager,
			embedder={
				"provider": "ollama",
				"config": {"model": "mxbai-embed-large"}
    				},
			)

		return crew_workflow
