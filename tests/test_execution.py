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
from networkx import MultiDiGraph
from copy import deepcopy
import warnings

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



def multidigraph_to_dict(G):
    """Convert a NetworkX MultiDiGraph to a dictionary.
    
    This function extracts all graph metadata, nodes, and edges (with their keys)
    into a serializable dictionary format.
    
    Parameters
    ----------
    G : MultiDiGraph
        The graph to convert
    
    Returns
    -------
    dict
        A dictionary representation of the graph with the following structure:
        {
            'directed': True,  # Always true for MultiDiGraph
            'multigraph': True,  # Always true for MultiDiGraph
            'graph': dict,  # Graph attributes
            'nodes': list of dicts,  # Node attributes
            'edges': list of dicts,  # Edge attributes with source, target, key
        }
    """
    if not isinstance(G, MultiDiGraph):
        raise TypeError("Input graph must be a NetworkX MultiDiGraph")
    
    # Initialize the dictionary with graph metadata
    data = {
        'directed': True,  # MultiDiGraph is always directed
        'multigraph': True,  # MultiDiGraph is always a multigraph
        'graph': deepcopy(G.graph),  # Graph attributes
        'nodes': [],
        'edges': []
    }
    
    # Add nodes with their attributes
    for node, node_attrs in G.nodes(data=True):
        node_dict = {'id': node}
        node_dict.update(deepcopy(node_attrs))
        data['nodes'].append(node_dict)
    
    # Add edges with their attributes and keys
    for u, v, key, edge_attrs in G.edges(data=True, keys=True):
        edge_dict = {
            'source': u,
            'target': v,
            'key': key
        }
        edge_dict.update(deepcopy(edge_attrs))
        data['edges'].append(edge_dict)
    
    return data


def dict_to_multidigraph(data):
    """Convert a dictionary representation back to a NetworkX MultiDiGraph.
    
    Parameters
    ----------
    data : dict
        Dictionary containing graph data with nodes, edges, and attributes
    
    Returns
    -------
    G : MultiDiGraph
        A MultiDiGraph reconstructed from the dictionary
    """
    # Validate the input data
    if not isinstance(data, dict):
        raise TypeError("Input data must be a dictionary")
    
    # Check if the data represents a multigraph and directed graph
    is_directed = data.get('directed', True)
    is_multigraph = data.get('multigraph', True)
    
    if not (is_directed and is_multigraph):
        # If the data doesn't indicate it's a directed multigraph, warn but continue
        import warnings
        warnings.warn("Input data does not specify a directed multigraph, but forcing creation of MultiDiGraph")
    
    # Create a new MultiDiGraph
    G = MultiDiGraph()
    
    # Set graph attributes
    if 'graph' in data:
        G.graph.update(deepcopy(data['graph']))
    
    # Add nodes with attributes
    for node_data in data.get('nodes', []):
        # Copy the node data dict
        node_attrs = deepcopy(node_data)
        # Extract the ID (remove it from attributes)
        node_id = node_attrs.pop('id')
        # Add the node with its attributes
        G.add_node(node_id, **node_attrs)
    
    # Add edges with attributes and keys
    for edge_data in data.get('edges', []):
        # Copy the edge data dict
        edge_attrs = deepcopy(edge_data)
        # Extract source, target, and key
        source = edge_attrs.pop('source')
        target = edge_attrs.pop('target')
        edge_key = edge_attrs.pop('key', None)  # Key is optional
        
        # Add the edge with its key and attributes
        G.add_edge(source, target, key=edge_key, **edge_attrs)
    
    return G


def workflow_graph_to_dict(workflow_graph):
    """Convert a WorkFlowGraph to a dictionary with built-in Python types.
    
    This function converts a WorkFlowGraph and all its components (nodes, edges, etc.)
    into a fully serializable dictionary containing only built-in Python types.
    
    Parameters
    ----------
    workflow_graph : WorkFlowGraph
        The workflow graph to convert
    
    Returns
    -------
    dict
        A dictionary representation of the workflow graph with all components
        converted to built-in Python types
    """
    # First use the to_dict method from BaseModule
    workflow_dict = workflow_graph.to_dict(exclude_none=True, ignore=["graph"])
    
    # If the graph attribute exists and is a MultiDiGraph, convert it to dict
    if workflow_graph.graph is not None:
        if isinstance(workflow_graph.graph, dict):
            # The graph is already a dict
            workflow_dict["graph"] = workflow_graph.graph
        else:
            # Convert MultiDiGraph to dict
            workflow_dict["graph"] = multidigraph_to_dict(workflow_graph.graph)
    
    return workflow_dict


def dict_to_workflow_graph(workflow_dict):
    """Convert a dictionary back to a WorkFlowGraph.
    
    This function reconstructs a WorkFlowGraph from a dictionary representation.
    
    Parameters
    ----------
    workflow_dict : dict
        Dictionary representation of a workflow graph
    
    Returns
    -------
    WorkFlowGraph
        The reconstructed WorkFlowGraph
    """
    # Make a copy to avoid modifying the original dict
    workflow_data = deepcopy(workflow_dict)
    
    # If 'graph' key exists in the dictionary
    graph_data = workflow_data.pop("graph", None)
    
    # Use from_dict method of BaseModule to create the WorkFlowGraph
    workflow_graph = WorkFlowGraph.from_dict(workflow_data)
    
    # If graph data exists, convert it back to MultiDiGraph and set it
    if graph_data is not None:
        workflow_graph.graph = dict_to_multidigraph(graph_data)
    
    return workflow_graph


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
    4. Serialize and deserialize the workflow graph
    """
    print("\n=== Testing Workflow Generator ===")
    
    # Initialize LLM
    llm_config = OpenAILLMConfig(
        # model="gpt-3.5-turbo",
        model="gpt-4o-mini",
        openai_key=OPENAI_API_KEY,
        temperature=0.3,
        top_p=0.9,
        stream=True,
        output_response=True
    )
    llm = OpenAILLM(config=llm_config)
    
    # Create workflow generator
    workflow_generator = WorkFlowGenerator(llm=llm)
    
    # Define a complex goal that requires multiple steps
    goal = """
    Create a web application that allows users to ask questions and get respond using ChatGPT with a key.
    """
    
    # from pdb import set_trace; set_trace()
    print("\n=== Generating Workflow ===")
    print("Goal:", goal)
    workflow_graph = workflow_generator.generate_workflow(goal=goal)
    
    # Convert WorkFlowGraph to dictionary with built-in types
    print("\n=== Converting WorkFlowGraph to Dictionary ===")
    workflow_dict = workflow_graph_to_dict(workflow_graph)
    print(f"Workflow dict type: {type(workflow_dict)}")
    
    # Reconstruct WorkFlowGraph from dictionary
    print("\n=== Reconstructing WorkFlowGraph from Dictionary ===")
    reconstructed_workflow = dict_to_workflow_graph(workflow_dict)
    print(f"Reconstructed workflow type: {type(reconstructed_workflow)}")
    
    # Verify the reconstruction worked correctly
    print("\nWorkflow description after reconstruction:")
    print(reconstructed_workflow.get_workflow_description())
    
    print(f"\nOriginal nodes: {len(workflow_graph.nodes)}")
    print(f"Reconstructed nodes: {len(reconstructed_workflow.nodes)}")
    
    assert len(workflow_graph.nodes) == len(reconstructed_workflow.nodes), "Node count should match"
    assert len(workflow_graph.edges) == len(reconstructed_workflow.edges), "Edge count should match"
    
    from pdb import set_trace; set_trace()

    assert False, "Test completed - check the output"

