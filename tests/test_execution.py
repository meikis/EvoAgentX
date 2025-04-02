import pytest
import pytest_asyncio
import uuid
from urllib.parse import urljoin
import httpx
import os
import sys
import io
from typing import Union, Tuple

from unittest.mock import patch
from evoagentx.agents import Agent
from evoagentx.models.model_configs import OpenAILLMConfig
from evoagentx.app.services import AgentService
from evoagentx.app.db import Agent as AppAgent
from evoagentx.app.config import settings
from evoagentx.agents.agent_manager import AgentManager
from evoagentx.prompts.bolt_prompt_system import BOLT_PROMPT
from evoagentx.actions.agent_generation import AgentGeneration
from evoagentx.models.openai_model import OpenAILLM
from evoagentx.prompts.requirement_collection import REQUIREMENT_COLLECTION_PROMPT
from evoagentx.workflow.workflow import WorkFlow
from evoagentx.workflow.workflow_generator import WorkFlowGenerator
from evoagentx.workflow.workflow_graph import WorkFlowGraph, WorkFlowNode, Parameter, WorkFlowEdge
from evoagentx.core.base_config import Parameter
from evoagentx.actions.action import Action, ActionInput, ActionOutput
from evoagentx.core.message import Message, MessageType
from typing import Optional, List
import json
from pydantic import Field
from evoagentx.agents.agent_generator import AgentGenerator
from evoagentx.actions.agent_generation import AgentGenerationOutput
# API KEY for OpenAI models
OPENAI_API_KEY="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"


# Custom action for question answering
class QAInput(ActionInput):
    question: str = Field(default="", description="The question to be answered.")
    history: List[dict] = Field(default=[], description="The history of the conversation as a list of dictionaries with 'role' and 'content' fields.")

class QAOutput(ActionOutput):
    answer: str = Field(default="", description="The answer to the question.")
    history: List[dict] = Field(default=[], description="The updated history of the conversation as a list of dictionaries with 'role' and 'content' fields.")
    
class SimpleQAAction(Action):
    """A simple action for question answering."""
    
    def __init__(self, **kwargs):
        name = kwargs.pop("name") if "name" in kwargs else "answer_question"
        description = kwargs.pop("description", "An action that answers questions using an LLM")
        prompt = kwargs.pop("prompt", "You are a helpful assistant that provides accurate, concise answers to questions.")
        inputs_format = kwargs.pop("inputs_format", QAInput)
        outputs_format = kwargs.pop("outputs_format", QAOutput)
        super().__init__(name=name, description=description, prompt=prompt, inputs_format=inputs_format, outputs_format=outputs_format, **kwargs)
    
    def execute(self, llm, inputs=None, sys_msg=None, return_prompt=False, **kwargs):
        """Execute the QA action with the given inputs."""
        # Extract the question from inputs
        if not inputs:
            raise ValueError('The `inputs` to QA action is None or empty.')
            
        # Convert dictionary to QAInput object if needed
        if isinstance(inputs, dict):
            inputs = QAInput(**inputs)
            
        # Get question and history from the QAInput object
        question = inputs.question
        history = inputs.history
        
        # Print detailed information about inputs
        print(f"\n[{self.name}] INPUT:")
        print(f"  Question: {question}")
        print(f"  History: {history}")
        
        # Format history for the LLM if needed
        llm_history = history
        
        response = llm.generate(
            prompt=question,
            history=llm_history,
            parse_mode="json"
        )
        
        # Update history with the new question and answer
        updated_history = history.copy()
        # Add the assistant's response
        updated_history.append({"role": "assistant", "content": response.content})
        
        # Create QAOutput object
        output = QAOutput(
            answer=response.content,
            history=updated_history
        )
        
        # Print detailed information about outputs
        print(f"\n[{self.name}] OUTPUT:")
        print(f"  Answer: {response.content[:100]}..." if len(response.content) > 100 else f"  Answer: {response.content}")
        print(f"  Updated History: {output.history}")
        
        # Return appropriate output based on return_prompt parameter
        if return_prompt:
            return output, question
        else:
            return output



# @pytest.mark.asyncio
# async def test_local_simple_qa_workflow():
#     """
#     Demonstrates creating and executing a simple QA workflow locally without using the API.
    
#     This test:
#     1. Creates an agent with QA capabilities
#     2. Adds a QA action to the agent
#     3. Creates a workflow graph with a single task
#     4. Executes the workflow locally
#     5. Verifies the output
#     """
#     print("\n=== Testing Simple QA Workflow Locally ===")
    
#     # Step 1: Create the LLM
#     print("Creating LLM...")
#     llm_config = OpenAILLMConfig(
#         model="gpt-3.5-turbo", 
#         openai_key=OPENAI_API_KEY,
#         temperature=0.3,
#         max_tokens=500,
#         top_p=0.9
#     )
#     llm = OpenAILLM(config=llm_config)
    
#     # Step 2: Create an agent manager and add a QA agent
#     print("Creating agent manager and QA agent...")
#     agent_manager = AgentManager()
#     agent_manager.init_module()
    
#     # Create QA agent
#     agent_manager.add_agent({
#         "name": "QAAgent",
#         "description": "An agent that can answer questions based on provided content",
#         "prompt": "You are a helpful assistant that provides accurate, concise answers to questions.",
#         "llm_config": llm_config,
#     })
    
#     # Add QA action to the agent
#     qa_agent = agent_manager.get_agent("QAAgent")
#     qa_action = SimpleQAAction(name="answer_question")
#     qa_agent.add_action(qa_action)
    
#     qa_node_1_config = {
#         "name": "QuestionAnswering",
#         "description": "Answer the provided question",
#         "inputs": [Parameter(name="question", type="str", description="The question to be answered"), Parameter(name="history", type="list", description="The history of the conversation")],
#         "outputs": [Parameter(name="answer", type="str", description="The answer to the question"), Parameter(name="history", type="list", description="The history of the conversation")],
#         "agents": ["QAAgent"]
#     }
    
#     qa_node_1 = WorkFlowNode.from_dict(qa_node_1_config)
#     nodes = [qa_node_1]
#     qa_graph = WorkFlowGraph(
#         goal="Answer the user's question accurately and concisely",
#         nodes=nodes,
#         edges=[]
#     )
#     # qa_graph.display()
    
#     workflow = WorkFlow(
#         graph=qa_graph,
#         llm=llm,
#         agent_manager=agent_manager
#     )
    
    
#     workflow_input = {"question": "What is the capital of France?", "history": []} 
#     result = workflow.execute(inputs=workflow_input)
    
#     print("___________________")
#     print("Final result: ", result)
#     print("___________________")
#     assert False
    
    
# @pytest.mark.asyncio
# async def test_workflow():
#     """
#     Creates and tests a workflow with two connected nodes:
#     1. First node: Takes a question and generates an initial answer
#     2. Second node: Takes the initial answer and refines/improves it
#     """
#     print("\n=== Testing Two-Node Workflow ===")
    
#     # Step 1: Create the LLM
#     print("Creating LLM...")
#     llm_config = OpenAILLMConfig(
#         model="gpt-3.5-turbo", 
#         openai_key=OPENAI_API_KEY,
#         temperature=0.3,
#         max_tokens=500,
#         top_p=0.9
#     )
#     llm = OpenAILLM(config=llm_config)
    
#     # Step 2: Create agent manager and two agents
#     print("Creating agent manager and agents...")
#     agent_manager = AgentManager()
#     agent_manager.init_module()
    
#     # Create QA agent
#     agent_manager.add_agent({
#         "name": "InitialQAAgent",
#         "description": "An agent that generates initial answers to questions",
#         "prompt": "You are a helpful assistant that provides accurate, concise answers to questions. Provide only basic information without elaboration.",
#         "llm_config": llm_config,
#     })
    
#     # Create answer refinement agent
#     agent_manager.add_agent({
#         "name": "RefinementAgent",
#         "description": "An agent that refines and improves answers",
#         "prompt": "You are a helpful assistant that improves and refines existing answers to make them more accurate, complete, and detailed.",
#         "llm_config": llm_config,
#     })
    
#     # Step 3: Create QA actions and add them to agents
#     initial_qa_agent = agent_manager.get_agent("InitialQAAgent")
#     initial_qa_action = SimpleQAAction(name="generate_initial_answer")
#     initial_qa_agent.add_action(initial_qa_action)
    
#     # Print available actions for the InitialQAAgent
#     print("\n--- InitialQAAgent Available Actions ---")
#     print(f"Action Map: {initial_qa_agent._action_map.keys()}")
#     for action in initial_qa_agent.actions:
#         print(f"Action: {action.name}, Type: {type(action).__name__}")
    
#     refinement_agent = agent_manager.get_agent("RefinementAgent")
#     refinement_qa_action = SimpleQAAction(name="refine_answer")
#     refinement_agent.add_action(refinement_qa_action)
    
#     # Print available actions for the RefinementAgent
#     print("\n--- RefinementAgent Available Actions ---")
#     print(f"Action Map: {refinement_agent._action_map.keys()}")
#     for action in refinement_agent.actions:
#         print(f"Action: {action.name}, Type: {type(action).__name__}")
    
#     # Step 4: Create workflow nodes
#     initial_node_config = {
#         "name": "InitialAnswering",
#         "description": "Generate an initial answer to the question",
#         "inputs": [Parameter(name="question", type="str", description="The question to be answered"), 
#                   Parameter(name="history", type="list", description="The history of the conversation as a list of dictionaries with 'role' and 'content' fields")],
#         "outputs": [Parameter(name="answer", type="str", description="The initial answer to the question"),
#                    Parameter(name="history", type="list", description="The updated history of the conversation as a list of dictionaries with 'role' and 'content' fields")],
#         "agents": ["InitialQAAgent"]
#     }
    
#     refinement_node_config = {
#         "name": "AnswerRefinement",
#         "description": "Refine and improve the initial answer",
#         "inputs": [Parameter(name="question", type="str", description="The original question"),
#                   Parameter(name="answer", type="str", description="The initial answer to refine"),
#                   Parameter(name="history", type="list", description="The history of the conversation")],
#         "outputs": [Parameter(name="answer", type="str", description="The refined answer"),
#                    Parameter(name="history", type="list", description="The updated history of the conversation")],
#         "agents": ["RefinementAgent"],
#         "action": "refine_answer"
#     }
    
#     # Create WorkFlow nodes from configs
#     initial_qa_node = WorkFlowNode.from_dict(initial_node_config)
#     refinement_node = WorkFlowNode.from_dict(refinement_node_config)
    
#     # Step 5: Create workflow graph with connected nodes
#     nodes = [initial_qa_node, refinement_node]
    
#     # Create an edge connecting the initial node to the refinement node
#     edge = WorkFlowEdge(edge_tuple=(initial_qa_node.name, refinement_node.name))
    
#     workflow_graph = WorkFlowGraph(
#         goal="Answer the user's question accurately and provide a refined, high-quality response",
#         nodes=nodes,
#         edges=[edge]
#     )
    
#     # Step 6: Create and execute workflow
#     workflow = WorkFlow(
#         graph=workflow_graph,
#         llm=llm,
#         agent_manager=agent_manager
#     )
    
#     # Execute the workflow with a more complex question that would benefit from refinement
#     complex_question = "What are the main differences between machine learning and deep learning?"
#     workflow_input = {
#         "question": complex_question, 
#         "history": []  # Empty history to start with, will be properly formatted as [{"role": "user", "content": "..."}, ...]
#     }
    
#     # Print detailed agent information before workflow execution
#     print("\n=== AGENT CONFIGURATION BEFORE WORKFLOW EXECUTION ===")
#     for agent_name in ["InitialQAAgent", "RefinementAgent"]:
#         agent = agent_manager.get_agent(agent_name)
#         print(f"\nAgent: {agent.name}")
#         print(f"Description: {agent.description}")
#         print("Actions:")
#         for action in agent.actions:
#             print(f"  - {action.name}: {action.description}")
    
#     print("\n=== STARTING WORKFLOW EXECUTION ===")
#     print(f"Question: {complex_question}")
#     print(f"Initial History: {workflow_input['history']}")
#     print("\n=== EXECUTING WORKFLOW ===")
    
#     result = workflow.execute(inputs=workflow_input)
    
#     print("\n=== WORKFLOW FINAL RESULT ===")
#     print(result)
    
#     # Verify the result contains key terms related to the question
#     assert "machine learning" in result.lower(), "Result should mention machine learning"
#     assert "deep learning" in result.lower(), "Result should mention deep learning"
#     assert len(result) > 100, "Result should be a detailed response with significant length"
    
#     print("\n=== TEST VERIFICATION ===")
#     print("✓ Result mentions 'machine learning'")
#     print("✓ Result mentions 'deep learning'")
#     print(f"✓ Result length ({len(result)} chars) is > 100 chars")
    
#     assert False


### Return/Save new agents / workflows
#### Create new agents everytime
#### Retrival --> DB

## Workflow creation start: save config to db
## Execution: execute agent -> save result to db

## FUture:
    # Local DB

## TODO:
    # Execution
    # DB save/load --> API
    # Workflow execution with DB



# @pytest.mark.asyncio
# async def test_workflow_with_api():
#     """
#     Creates and tests a workflow with two connected nodes:
#     1. First node: Takes a question and generates an initial answer
#     2. Second node: Takes the initial answer and refines/improves it
#     """
#     print("\n=== Testing Workflow API Configuration ===")
    
#     # Define the complete workflow configuration
#     workflow_config = {
#         "workflow": {
#             "name": "Two-Stage QA Workflow",
#             "description": "A workflow that generates and refines answers to questions",
#             "goal": "To provide comprehensive, accurate answers to user questions",
#             "version": "1.0"
#         },
#         "llm_config": {
#             "provider": "openai",
#             "model": "gpt-3.5-turbo",
#             "openai_key": OPENAI_API_KEY,
#             "temperature": 0.3,
#             "max_tokens": 500,
#             "top_p": 0.9
#         },
#         ## Agent creation or calling
#         "agents": [
#             {
#                 "name": "InitialQAAgent",
#                 "description": "An agent that generates initial answers to questions",
#                 "prompt": "You are a helpful assistant that provides accurate, concise answers to questions. Provide only basic information without elaboration.",
#                 "llm_config": "default"  # Uses the default LLM config defined above
#             },
#             {
#                 "name": "RefinementAgent",
#                 "description": "An agent that refines and improves answers",
#                 "prompt": "You are a helpful assistant that improves and refines existing answers to make them more accurate, complete, and detailed.",
#                 "llm_config": "default"  # Uses the default LLM config defined above
#             }
#         ],
#         ## 
#         "actions": [
#             {
#                 "name": "generate_initial_answer",
#                 "type": "SimpleQAAction",
#                 "description": "Generates an initial answer to a question",
#                 "prompt": "Provide a concise answer to the question",
#                 "agent": "InitialQAAgent"
#             },
#             {
#                 "name": "refine_answer",
#                 "type": "SimpleQAAction",
#                 "description": "Refines an existing answer to make it more detailed",
#                 "prompt": "Enhance this answer with more details and examples",
#                 "agent": "RefinementAgent"
#             }
#         ],
#         "nodes": [
#             {
#                 "name": "InitialAnswering",
#                 "description": "Generate an initial answer to the question",
#                 "agents": ["InitialQAAgent"],
#                 "action": "generate_initial_answer",
#                 "inputs": [
#                     {"name": "question", "type": "str", "description": "The question to be answered"},
#                     {"name": "history", "type": "list", "description": "The conversation history"}
#                 ],
#                 "outputs": [
#                     {"name": "answer", "type": "str", "description": "The initial answer"},
#                     {"name": "history", "type": "list", "description": "The updated conversation history"}
#                 ]
#             },
#             {
#                 "name": "AnswerRefinement",
#                 "description": "Refine and improve the initial answer",
#                 "agents": ["RefinementAgent"],
#                 "action": "refine_answer",
#                 "inputs": [
#                     {"name": "question", "type": "str", "description": "The original question"},
#                     {"name": "answer", "type": "str", "description": "The initial answer to refine"},
#                     {"name": "history", "type": "list", "description": "The conversation history"}
#                 ],
#                 "outputs": [
#                     {"name": "answer", "type": "str", "description": "The refined answer"},
#                     {"name": "history", "type": "list", "description": "The updated conversation history"}
#                 ]
#             }
#         ],
#         "edges": [
#             {
#                 "source": "InitialAnswering",
#                 "target": "AnswerRefinement",
#                 "mappings": [
#                     {"source_output": "question", "target_input": "question"},
#                     {"source_output": "answer", "target_input": "answer"},
#                     {"source_output": "history", "target_input": "history"}
#                 ]
#             }
#         ]
#     }
    
#     print("Workflow configuration defined successfully")
#     print(f"Number of agents: {len(workflow_config['agents'])}")
#     print(f"Number of actions: {len(workflow_config['actions'])}")
#     print(f"Number of nodes: {len(workflow_config['nodes'])}")
#     print(f"Number of edges: {len(workflow_config['edges'])}")
    
#     # Implement the workflow creation from configuration
#     print("\n=== Creating Workflow From Configuration ===")
    
#     # Step 1: Create the LLM from llm_config
#     print("Creating LLM...")
#     llm_config_dict = workflow_config["llm_config"]
#     llm_config = OpenAILLMConfig(
#         model=llm_config_dict["model"],
#         openai_key=llm_config_dict["openai_key"],
#         temperature=llm_config_dict["temperature"],
#         max_tokens=llm_config_dict["max_tokens"],
#         top_p=llm_config_dict["top_p"]
#     )
#     llm = OpenAILLM(config=llm_config)
    
#     # Step 2: Create agent manager and add agents
#     print("Creating agent manager and adding agents...")
#     agent_manager = AgentManager()
#     agent_manager.init_module()
    
#     # Create agents from configuration
#     for agent_config in workflow_config["agents"]:
#         # If llm_config is "default", use the default one
#         if agent_config["llm_config"] == "default":
#             agent_llm_config = llm_config
        
#         agent_manager.add_agent({
#             "name": agent_config["name"],
#             "description": agent_config["description"],
#             "prompt": agent_config["prompt"],
#             "llm_config": agent_llm_config,
#         })
#         print(f"Added agent: {agent_config['name']}")
    
#     # Step 3: Create actions and assign to agents
#     print("\nCreating actions and assigning to agents...")
    
#     # Mapping of action types to action classes
#     action_types = {
#         "SimpleQAAction": SimpleQAAction
#     }
    
#     # Create actions and add to agents
#     for action_config in workflow_config["actions"]:
#         if action_config["type"] in action_types:
#             action_class = action_types[action_config["type"]]
#             action_instance = action_class(
#                 name=action_config["name"],
#                 description=action_config.get("description", ""),
#                 prompt=action_config.get("prompt", "")
#             )
            
#             # Get the agent and add the action
#             agent = agent_manager.get_agent(action_config["agent"])
#             if agent:
#                 agent.add_action(action_instance)
#                 print(f"Added action '{action_config['name']}' to agent '{action_config['agent']}'")
#             else:
#                 print(f"Error: Agent '{action_config['agent']}' not found for action '{action_config['name']}'")
    
#     # Step 4: Create workflow nodes
#     print("\nCreating workflow nodes...")
#     nodes = []
    
#     # Convert node configurations to actual WorkFlowNode objects
#     for node_config in workflow_config["nodes"]:
#         # Create Parameter objects for inputs
#         inputs = [
#             Parameter(
#                 name=input_param["name"],
#                 type=input_param["type"],
#                 description=input_param["description"]
#             ) for input_param in node_config["inputs"]
#         ]
        
#         # Create Parameter objects for outputs
#         outputs = [
#             Parameter(
#                 name=output_param["name"],
#                 type=output_param["type"],
#                 description=output_param["description"]
#             ) for output_param in node_config["outputs"]
#         ]
        
#         # Create a dictionary for node configuration
#         node_dict = {
#             "name": node_config["name"],
#             "description": node_config["description"],
#             "inputs": inputs,
#             "outputs": outputs,
#             "agents": node_config["agents"]
#         }
        
#         # Add action if specified
#         if "action" in node_config:
#             node_dict["action"] = node_config["action"]
        
#         # Create the WorkFlowNode from the dictionary
#         workflow_node = WorkFlowNode.from_dict(node_dict)
#         nodes.append(workflow_node)
#         print(f"Created node: {node_config['name']}")
    
#     # Step 5: Create edges between nodes
#     print("\nCreating workflow edges...")
#     edges = []
    
#     # Create a dictionary to quickly look up nodes by name
#     node_map = {node.name: node for node in nodes}
    
#     # Create WorkFlowEdge objects from edge configurations
#     for edge_config in workflow_config["edges"]:
#         source_node_name = edge_config["source"]
#         target_node_name = edge_config["target"]
        
#         # Check if both source and target nodes exist
#         if source_node_name in node_map and target_node_name in node_map:
#             # Create the edge
#             edge = WorkFlowEdge(edge_tuple=(source_node_name, target_node_name))
#             edges.append(edge)
#             print(f"Created edge: {source_node_name} -> {target_node_name}")
            
#             # Print the mappings (not used in edge creation but useful for debugging)
#             if "mappings" in edge_config:
#                 for mapping in edge_config["mappings"]:
#                     print(f"  Mapping: {mapping['source_output']} -> {mapping['target_input']}")
#         else:
#             print(f"Error: Cannot create edge. Source or target node not found: {source_node_name} -> {target_node_name}")
    
#     # Step 6: Assemble the workflow graph
#     print("\nAssembling workflow graph...")
#     workflow_graph = WorkFlowGraph(
#         goal=workflow_config["workflow"]["goal"],
#         nodes=nodes,
#         edges=edges
#     )
#     print(f"Created workflow graph with {len(nodes)} nodes and {len(edges)} edges")
    
#     # Step 7: Create and execute the workflow
#     print("\nCreating workflow...")
#     workflow = WorkFlow(
#         graph=workflow_graph,
#         llm=llm,
#         agent_manager=agent_manager
#     )
#     print("Workflow created successfully")
    
#     # Execute the workflow with a test question
#     print("\n=== EXECUTING WORKFLOW ===")
#     complex_question = "What are the main differences between machine learning and deep learning?"
#     workflow_input = {
#         "question": complex_question, 
#         "history": []  # Empty history to start with
#     }
    
#     print(f"Question: {complex_question}")
#     result = workflow.execute(inputs=workflow_input)
    
#     print("\n=== WORKFLOW FINAL RESULT ===")
#     print(result)
    
#     # Verify the result contains key terms related to the question
#     assert "machine learning" in result.lower(), "Result should mention machine learning"
#     assert "deep learning" in result.lower(), "Result should mention deep learning"
#     assert len(result) > 100, "Result should be a detailed response with significant length"
    
#     print("\n=== TEST VERIFICATION ===")
#     print("✓ Result mentions 'machine learning'")
#     print("✓ Result mentions 'deep learning'")
#     print(f"✓ Result length ({len(result)} chars) is > 100 chars")
    
#     # For debugging, force test to stop here
#     print("\nTest completed successfully - stopping for analysis")
    
#     assert False


# @pytest.mark.asyncio
# async def test_workflow_generator():
#     """
#     Test the workflow generator's ability to:
#     1. Generate a task plan
#     2. Build a workflow graph
#     3. Generate appropriate agents for tasks
#     """
#     print("\n=== Testing Workflow Generator ===")
    
#     # Initialize LLM
#     llm_config = OpenAILLMConfig(
#         model="gpt-4",
#         openai_key=OPENAI_API_KEY,
#         temperature=0.3,
#         max_tokens=1000,
#         top_p=0.9
#     )
#     llm = OpenAILLM(config=llm_config)
    
#     # Create workflow generator
#     workflow_generator = WorkFlowGenerator(llm=llm)
    
#     # Define a complex goal that requires multiple steps
#     goal = """
#     Create a web application that allows users to ask questions and get respond using ChatGPT with a key.
#     """
    
#     print("\n=== Generating Workflow ===")
#     print("Goal:", goal)
    
#     # Generate the workflow
#     workflow = workflow_generator.generate_workflow(goal=goal)
    
#     # Verify workflow structure
#     print("\n=== Verifying Workflow Structure ===")
#     print("Workflow Goal:", workflow.goal)
#     print("\nNodes:")
#     for node in workflow.nodes:
#         print(f"\nNode: {node.name}")
#         print(f"Description: {node.description}")
#         print("Inputs:", [f"{p.name} ({p.type})" for p in node.inputs])
#         print("Outputs:", [f"{p.name} ({p.type})" for p in node.outputs])
#         print("Assigned Agents:", node.get_agents())
    
#     print("\nEdges:")
#     for edge in workflow.edges:
#         print(f"{edge.source.name} -> {edge.target.name}")
    
#     # Verify that the workflow has essential components
#     assert workflow.goal == goal, "Workflow goal should match input goal"
#     assert len(workflow.nodes) > 0, "Workflow should have at least one node"
#     assert all(node.get_agents() for node in workflow.nodes), "All nodes should have assigned agents"
    
#     # Verify workflow connectivity
#     node_names = {node.name for node in workflow.nodes}
#     edge_sources = {edge.source.name for edge in workflow.edges}
#     edge_targets = {edge.target.name for edge in workflow.edges}
    
#     # Check that all nodes (except start and end nodes) are connected
#     middle_nodes = node_names - (node_names - edge_targets) - (node_names - edge_sources)
#     assert len(middle_nodes) > 0, "Workflow should have connected nodes"
    
#     # Check that each middle node has both incoming and outgoing edges
#     for node_name in middle_nodes:
#         assert node_name in edge_sources, f"Node {node_name} should have outgoing edges"
#         assert node_name in edge_targets, f"Node {node_name} should have incoming edges"
    
#     print("\n=== Workflow Generation Test Complete ===")
#     assert False

# @pytest.mark.asyncio
# async def test_llm_parsing_for_task_planning():
#     """
#     Simple test to generate and parse LLM response for TaskPlanningOutput.
#     Focus on understanding the validation error for sub_tasks field.
#     """
#     print("\n=== Testing LLM Response for TaskPlanningOutput ===")
    
#     # Initialize LLM with the same config as in workflow_generator test
#     llm_config = OpenAILLMConfig(
#         model="gpt-4", 
#         openai_key=OPENAI_API_KEY,
#         temperature=0.3,
#         max_tokens=2000,  # Increased to avoid truncation
#         top_p=0.9
#     )
#     llm = OpenAILLM(config=llm_config)
    
#     # Import necessary classes
#     from evoagentx.prompts.task_planner import TASK_PLANNING_ACTION
#     from evoagentx.actions.task_planning import TaskPlanningOutput, TaskPlanning
#     from evoagentx.core.message import Message, MessageType
#     from evoagentx.workflow.workflow_generator import TaskPlanner
#     # Use the same goal as in the original test
#     goal = "Create a web application that allows users to ask questions and get respond using ChatGPT with a key."
    
#     task_planner = TaskPlanner(llm=llm)
#     task_planning_action_data = {"goal": goal, "history": [], "suggestion": []}
#     task_planning_action_name = task_planner.task_planning_action_name
#     message: Message = task_planner.execute(
#         action_name=task_planning_action_name,
#         action_input_data=task_planning_action_data,
#         return_msg_type=MessageType.REQUEST
#     )
#     # return message.content
#     content_str = str(message.content)
#     plan = TaskPlanningOutput.parse(content_str, parse_mode="json")
#     print(f"_________________________")
#     print(f"type of plan: {type(plan)}")
#     print("\nPlan:\n", plan)
#     print(f"_________________________")
#     print("type of sub_tasks: ", type(plan.sub_tasks))
#     print("sub_tasks: ", plan.sub_tasks)
#     print(f"_________________________")
    

    
#     # # Use the actual prompt from task planner
#     # prompt = TASK_PLANNING_ACTION["prompt"].format(
#     #     goal=goal,
#     #     history="",
#     #     suggestion=""
#     # )
    
#     # print(f"Goal: {goal}")
    
#     # # Generate response from LLM
#     # print("\n=== Generating LLM Response ===")
#     # response = llm.generate(
#     #     prompt=prompt,
#     #     parse_mode="json"
#     # )
    
#     # # Print the raw response
#     # print("\n=== Raw LLM Response ===")
#     # print(f"Type: {type(response)}")
#     # print(f"Raw LLM Response: {response}")
#     # print(f"Content type: {type(response.content)}")
    
#     # # Extract JSON from the response
#     # print("\n=== Extracting JSON from Response ===")
#     # content_str = str(response.content)
    
#     # # Use the built-in parse method from LLMOutputParser
#     # plan = TaskPlanningOutput.parse(content_str, parse_mode="json")
#     # print(f"_________________________")
#     # print(f"type of plan: {type(plan)}")
#     # print("\nExtracted potential JSON content:\n", plan)
#     # print(f"_________________________")
#     # print(plan.sub_tasks)
    
    
    
#     assert False, "Test completed - check the output"




@pytest.mark.asyncio
async def test_workflow_generator_2():
    """
    Test the workflow generator's ability to:
    1. Generate a task plan
    2. Build a workflow graph
    3. Generate appropriate agents for tasks
    """
    print("\n=== Testing Workflow Generator ===")
    
    # Initialize LLM
    llm_config = OpenAILLMConfig(
        model="gpt-4",
        openai_key=OPENAI_API_KEY,
        temperature=0.3,
        max_tokens=1000,
        top_p=0.9
    )
    llm = OpenAILLM(config=llm_config)
    
    # Create workflow generator
    workflow_generator = WorkFlowGenerator(llm=llm)
    
    # Define a complex goal that requires multiple steps
    goal = """
    Create a web application that allows users to ask questions and get respond using ChatGPT with a key.
    """
    
    print("\n=== Generating Workflow ===")
    print("Goal:", goal)
    
    # Generate the workflow
    plan_history, plan_suggestion = "", ""
    # generate the initial workflow
    print("Generating the initial workflow ...")
    plan = workflow_generator.generate_plan(goal=goal, history=plan_history, suggestion=plan_suggestion)
    print(f"_________________________")
    print(f"type of plan: {type(plan)}")
    print("\nJSON content:\n", plan)
    print(f"_________________________")
    print(plan.sub_tasks)z
    
    print("Building the workflow from the plan ...")
    workflow = workflow_generator.build_workflow_from_plan(goal=goal, plan=plan)
    print(f"_________________________")
    print(f"type of workflow: {type(workflow)}")
    print("\nWorkflow content:\n", workflow)
    print(f"_________________________")
    
    print("Generating agents for the workflow ...")
    print("workflow_desc: ", workflow.get_workflow_description())
    workflow = workflow_generator.generate_agents(goal=goal, workflow=workflow, existing_agents=[])
    print(f"_________________________")
    print(f"type of workflow: {type(workflow)}")
    print("\nWorkflow content:\n", workflow)
    print(f"_________________________")
    # workflow = {"class_name": "WorkFlowGraph", "goal": "\n    Create a web application that allows users to ask questions and get respond using ChatGPT with a key.\n    ", "nodes": [{"class_name": "WorkFlowNode", "name": "design_interface", "description": "Design the user interface for the web application.", "inputs": [{"class_name": "Parameter", "name": "goal", "type": "string", "description": "The user's goal in textual format.", "required": true}], "outputs": [{"class_name": "Parameter", "name": "interface_design", "type": "string", "description": "The design of the user interface for the web application.", "required": true}], "reason": "A user-friendly interface is necessary for users to interact with the application.", "status": "pending"}, {"class_name": "WorkFlowNode", "name": "setup_backend", "description": "Set up the backend for the web application.", "inputs": [{"class_name": "Parameter", "name": "interface_design", "type": "string", "description": "The design of the user interface for the web application.", "required": true}], "outputs": [{"class_name": "Parameter", "name": "backend_setup", "type": "string", "description": "The setup of the backend for the web application.", "required": true}], "reason": "The backend is necessary to handle user requests and responses.", "status": "pending"}, {"class_name": "WorkFlowNode", "name": "integrate_chatgpt", "description": "Integrate the ChatGPT API into the web application.", "inputs": [{"class_name": "Parameter", "name": "backend_setup", "type": "string", "description": "The setup of the backend for the web application.", "required": true}], "outputs": [{"class_name": "Parameter", "name": "chatgpt_integration", "type": "string", "description": "The integration of the ChatGPT API into the web application.", "required": true}], "reason": "The ChatGPT API is necessary to generate responses to user questions.", "status": "pending"}, {"class_name": "WorkFlowNode", "name": "test_application", "description": "Test the web application to ensure it works as expected.", "inputs": [{"class_name": "Parameter", "name": "chatgpt_integration", "type": "string", "description": "The integration of the ChatGPT API into the web application.", "required": true}], "outputs": [{"class_name": "Parameter", "name": "test_results", "type": "string", "description": "The results of testing the web application.", "required": true}], "reason": "Testing is necessary to identify and fix any issues before the application is released.", "status": "pending"}], "edges": [{"class_name": "WorkFlowEdge", "source": "design_interface", "target": "setup_backend", "priority": 0}, {"class_name": "WorkFlowEdge", "source": "setup_backend", "target": "integrate_chatgpt", "priority": 0}, {"class_name": "WorkFlowEdge", "source": "integrate_chatgpt", "target": "test_application", "priority": 0}], "graph": "<networkx.classes.multidigraph.MultiDiGraph object at 0x7ff138d9b100>"}
    # workflow_desc = workflow.get_workflow_description()
    # agent_generator = AgentGenerator(llm=llm)
    # for subtask in workflow.nodes:
    #         subtask_fields = ["name", "description", "reason", "inputs", "outputs"]
    #         subtask_data = {key: value for key, value in subtask.to_dict(ignore=["class_name"]).items() if key in subtask_fields}
    #         subtask_desc = json.dumps(subtask_data, indent=4)
    #         agent_generation_action_data = {"goal": goal, "workflow": workflow_desc, "task": subtask_desc}
    #         agents: AgentGenerationOutput = agent_generator.execute(
    #             action_name=AgentGenerator.agent_generation_action_name, 
    #             action_input_data=agent_generation_action_data,
    #             return_msg_type=MessageType.RESPONSE
    #         ).content
    #         # todo I only handle generated agents
    #         generated_agents = []
    #         for agent in agents.generated_agents:
    #             agent_dict = agent.to_dict(ignore=["class_name"])
    #             agent_dict["llm_config"] = llm.config.to_dict()
    #             generated_agents.append(agent_dict)
    #         subtask.set_agents(agents=generated_agents)

    assert False, "Test completed - check the output"

    