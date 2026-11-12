from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from ai_agent_local_llm.tools.custom_tool import Input_data,Init_Pysr,Make_Prompt_From_Tools

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class AiAgentLocalLlm():
    """AiAgentLocalLlm crew 数据分析agent 根据真实数据总结出规律并返回"""

    agents: List[BaseAgent]
    tasks: List[Task]
   
    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def data_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['data_analyst'], # type: ignore[index]
            tools = [Input_data],
            verbose=True
        )    
    # @agent
    # def generate_f(self) -> Agent:
    #     return Agent(
    #         config=self.agents_config['generate_f'], # type: ignore[index]
    #         # tools = [Input_data],
    #         verbose=True
    #     )    
    # @agent
    # def init_pysr_analyst(self) -> Agent:
    #     return Agent(
    #         config=self.agents_config['init_pysr_analyst'], # type: ignore[index]
    #         tools = [Init_Pysr],
    #         verbose=True
    #     )    
    # @agent
    # def combined_agent(self) -> Agent:
    #     return Agent(
    #         config=self.agents_config['combined_agent'],
    #         # tools=[Input_data, Init_Pysr, Make_Prompt_From_Tools],
    #         verbose=True
    # )
    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def data_analyst_task(self) -> Task:
        return Task(
            config=self.tasks_config['data_analyst_task'], # type: ignore[index]
            tools = [Input_data],
            # async_execution=True # 将异步执行

            # output_key="data_analysis_result",  # 给结果起个名字
            output_file='report1.txt'
        )
    # print(data_analyst_task().output)

    # @task
    # def init_pysr_analyst_task(self) -> Task:
    #     return Task(
    #         config=self.tasks_config['init_pysr_analyst_task'], # type: ignore[index]
    #         tools = [Init_Pysr],
    #         # async_execution=True # 将异步执行
    #         # context=["data_analysis_result"],  # 依赖上一个任务的结果
    #         output_file='report2.txt'
    #     )
    # print(init_pysr_analyst_task().output)
    # @task
    # def make_prompt_task(self) -> Task:
    #     # print("DEBUG make_prompt_task config:", type(self.tasks_config['make_prompt_task']), self.tasks_config['make_prompt_task'])
    #     return Task(
    #         config=self.tasks_config['make_prompt_task'], # type: ignore[index]
    #         # tools=[Make_Prompt_From_Tools],
    #         context=[self.data_analyst_task(),self.init_pysr_analyst_task()],  # 自动接收上游结果
    #         # output_key="final_prompt"
    #         output_file='report3.txt'
    #     )
    @crew
    def crew(self) -> Crew:
        """Creates the AiAgentLocalLlm crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge
        # print(self.tasks.out)
        return Crew(
            agents=self.data_analyst(), # Automatically created by the @agent decorator
            tasks=self.data_analyst_task(), # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
@CrewBase
class Generate_function():
    """Generate_function crew"""
    agents: List[BaseAgent]
    tasks: List[Task]
    @agent
    def generate_f(self) -> Agent:
        return Agent(
            config=self.agents_config['generate_f'], # type: ignore[index]
            # tools = [Input_data],
            verbose=True
        )    
    @task
    def generate_f_task(self) -> Task:
        return Task(
            config=self.tasks_config['generate_f_task'], # type: ignore[index]
            # tools = [Input_data],
            # async_execution=True # 将异步执行
            # output_key="data_analysis_result",  # 给结果起个名字
            # output_file='report2.txt'
            verbose = True
        )
    @crew
    def crew(self) -> Crew:
        """Creates the AiAgentLocalLlm crew"""
        return Crew(
            agents=self.generate_f(), # Automatically created by the @agent decorator
            tasks=self.generate_f_task(), # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
       
