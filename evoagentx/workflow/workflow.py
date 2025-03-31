import inspect
from pydantic import Field
from typing import Optional
from ..core.logging import logger
from ..core.module import BaseModule
from ..core.message import Message, MessageType
from ..core.module_utils import generate_id
from ..models.base_model import BaseLLM
from ..agents.agent import Agent
from ..agents.agent_manager import AgentManager, AgentState
from ..storages.base import StorageHandler
from .environment import Environment, TrajectoryState
from .workflow_manager import WorkFlowManager, NextAction
from .workflow_graph import WorkFlowNode, WorkFlowGraph
from .action_graph import ActionGraph


class WorkFlow(BaseModule):

    graph: WorkFlowGraph
    llm: Optional[BaseLLM] = None
    agent_manager: AgentManager = Field(default=None, description="Responsible for managing agents")
    workflow_manager: WorkFlowManager = Field(default=None, description="Responsible for task and action scheduling for workflow execution")
    environment: Environment = Field(default_factory=Environment)
    storage_handler: StorageHandler = None
    workflow_id: str = Field(default_factory=generate_id)
    version: int = 0 

    def init_module(self):
        if self.workflow_manager is None:
            if self.llm is None:
                raise ValueError("Must provide `llm` when `workflow_manager` is None")
            self.workflow_manager = WorkFlowManager(llm=self.llm)

    def execute(self, inputs: dict = {}, **kwargs) -> str:

        goal = self.graph.goal
        inputs.update({"goal": goal})
        inp_message = Message(content=inputs, msg_type=MessageType.INPUT, wf_goal=goal)
        self.environment.update(message=inp_message, state=TrajectoryState.COMPLETED)

        failed = False
        while not self.graph.is_complete and not failed:
            try:
                task: WorkFlowNode = self.get_next_task()
                if task is None:
                    break
                logger.info(f"Executing subtask: {task.name}")
                self.execute_task(task=task)
            except Exception as e:
                failed = True
                error_message = Message(
                    content=f"An Error occurs when executing the workflow: {e}",
                    msg_type=MessageType.ERROR, 
                    wf_goal=goal
                )
                self.environment.update(message=error_message, state=TrajectoryState.FAILED, error=str(e))
        
        if failed:
            logger.error(error_message.content)
            return "Workflow Execution Failed"
        
        logger.info("Extracting WorkFlow Output ...")
        output: str = self.workflow_manager.extract_output(graph=self.graph, env=self.environment)
        return output
    
    def get_next_task(self) -> WorkFlowNode:
        task_execution_history = " -> ".join(self.environment.task_execution_history)
        if not task_execution_history:
            task_execution_history = "None"
        logger.info(f"Task Execution Trajectory: {task_execution_history}. Scheduling next subtask ...")
        task: WorkFlowNode = self.workflow_manager.schedule_next_task(graph=self.graph, env=self.environment)
        logger.info(f"The next subtask to be executed is: {task.name}")
        return task
        
    def execute_task(self, task: WorkFlowNode):

        last_executed_task = self.environment.get_last_executed_task()
        self.graph.step(source_node=last_executed_task, target_node=task)
        next_action: NextAction = self.workflow_manager.schedule_next_action(
            goal=self.graph.goal,
            task=task, 
            agent_manager=self.agent_manager, 
            env=self.environment
        )
        if next_action.action_graph is not None:
            self._execute_task_by_action_graph(task=task, next_action=next_action)
        else:
            self._execute_task_by_agents(task=task, next_action=next_action)
        self.graph.completed(node=task)

    def _execute_task_by_action_graph(self, task: WorkFlowNode, next_action: NextAction):

        action_graph: ActionGraph = next_action.action_graph
        execute_signature = inspect.signature(type(action_graph).execute)
        execute_params = []
        for param, _ in execute_signature.parameters.items():
            if param in ["self", "args", "kwargs"]:
                continue
            execute_params.append(param)
        action_input_data = self.environment.get_all_execution_data()
        execute_inputs = {param: action_input_data.get(param, "") for param in execute_params}
        action_graph_output: dict = action_graph.execute(**execute_inputs)
        message = Message(
            content=action_graph_output, action=action_graph.name, msg_type=MessageType.RESPONSE, \
                wf_goal=self.graph.goal, wf_task=task.name, wf_task_desc=task.description
        )
        self.environment.update(message=message, state=TrajectoryState.COMPLETED)
    
    def _execute_task_by_agents(self, task: WorkFlowNode, next_action: NextAction):
        
        while next_action:
            agent: Agent = self.agent_manager.get_agent(agent_name=next_action.agent)
            # while self.agent_manager.get_agent_state(agent_name=agent.name) != AgentState.AVAILABLE:
            #     sleep(5)
            if not self.agent_manager.wait_for_agent_available(agent_name=agent.name, timeout=300):
                raise TimeoutError(f"Timeout waiting for agent {agent.name} to become available")
            self.agent_manager.set_agent_state(agent_name=next_action.agent, new_state=AgentState.RUNNING)
            try:
                message = agent.execute(
                    action_name=next_action.action,
                    action_input_data=self.environment.get_all_execution_data(),
                    return_msg_type=MessageType.RESPONSE, 
                    wf_goal=self.graph.goal,
                    wf_task=task.name, 
                    wf_task_desc=task.description
                )
                self.environment.update(message=message, state=TrajectoryState.COMPLETED)
            finally:
                self.agent_manager.set_agent_state(agent_name=next_action.agent, new_state=AgentState.AVAILABLE)
            if self.is_task_completed(task=task):
                break
            next_action: NextAction = self.workflow_manager.schedule_next_action(
                goal=self.graph.goal,
                task=task,
                agent_manager=self.agent_manager, 
                env=self.environment
            )

    def is_task_completed(self, task: WorkFlowNode) -> bool:
        task_outputs = [output.name for output in task.outputs]
        current_execution_data = self.environment.get_all_execution_data()
        return all(output in current_execution_data for output in task_outputs)