import pytest
import pytest_asyncio
import uuid
from urllib.parse import urljoin
import httpx
import os
from evoagentx.agents import Agent
from evoagentx.models.model_configs import OpenAILLMConfig
from evoagentx.app.services import AgentService
from evoagentx.app.db import Agent as AppAgent
from evoagentx.app.config import settings
from evoagentx.agents.agent_manager import AgentManager
from evoagentx.prompts.bolt_prompt_system import BOLT_PROMPT
from evoagentx.actions.agent_generation import AgentGeneration
from evoagentx.models.openai_model import OpenAILLM


# Base API URL configuration
BASE_URL = "http://localhost:8000/api/v1/"
OPENAI_API_KEY="sample"

# --- Helper functions --- #
async def create_agent(client: httpx.AsyncClient, headers: dict, payload: dict) -> str:
    """Helper to create an agent and return its ID."""
    response = await client.post(urljoin(BASE_URL, "agents"), headers=headers, json=payload)
    print(f"Response: {response.json()}")
    assert response.status_code in (200, 201), f"Unexpected status: {response.status_code}"
    agent_data = response.json()
    return agent_data["_id"]

async def delete_agent(client: httpx.AsyncClient, headers: dict, agent_id: str):
    """Helper to delete an agent."""
    response = await client.delete(urljoin(BASE_URL, f"agents/{agent_id}"), headers=headers)
    assert response.status_code == 204

async def create_workflow(client: httpx.AsyncClient, headers: dict, payload: dict) -> str:
    """Helper to create a workflow and return its ID."""
    response = await client.post(urljoin(BASE_URL, "workflows"), headers=headers, json=payload)
    assert response.status_code == 201, f"Unexpected status: {response.status_code}"
    workflow_data = response.json()
    return workflow_data["_id"]

async def delete_workflow(client: httpx.AsyncClient, headers: dict, workflow_id: str):
    """Helper to delete a workflow."""
    response = await client.delete(urljoin(BASE_URL, f"workflows/{workflow_id}"), headers=headers)
    assert response.status_code == 204

# --- Fixtures --- #
@pytest_asyncio.fixture
async def client():
    async with httpx.AsyncClient() as client:
        yield client

@pytest_asyncio.fixture
async def access_token(client: httpx.AsyncClient):
    response = await client.post(
        urljoin(BASE_URL, "auth/login"),
        data={"username": "admin@clayx.ai", "password": "adminpassword"}
    )
    assert response.status_code == 200
    return response.json()["access_token"]

# # --- Test Cases --- #
# @pytest.mark.asyncio

@pytest.mark.asyncio
async def test_execute_agent(client: httpx.AsyncClient, access_token: str):
    headers = {"Authorization": f"Bearer {access_token}"}
    
    
    #### ___________ Configurations ___________ ####
    # OpenAI config
    config = {
        "llm_type": "OpenAILLM",
        "model": "gpt-3.5-turbo",
        "openai_key": OPENAI_API_KEY,
        "temperature": 0.7,
        "max_tokens": 150,
        "top_p": 0.9,
        "output_response": True,
        "prompt": "You are a helpful assistant that can help with a variety of tasks.",
    }

    
    # # For OpenAI through LiteLLM
    # config = {
    #     "llm_type": "LiteLLM",
    #     "model": "gpt-3.5-turbo",
    #     "openai_key": "your-openai-key",
    #     "temperature": 0.7,
    #     "output_response": True
    # }

    # # For Anthropic through LiteLLM
    # config = {
    #     "llm_type": "LiteLLM",
    #     "model": "anthropic/claude-2",
    #     "anthropic_key": "your-anthropic-key",
    #     "temperature": 0.7,
    #     "output_response": True
    # }

    # # For DeepSeek through LiteLLM
    # config = {
    #     "llm_type": "LiteLLM",
    #     "model": "deepseek-ai/deepseek-chat",
    #     "deepseek_key": "your-deepseek-key",
    #     "temperature": 0.7,
    #     "output_response": True
    # }
    
    # config = {
    #     "llm_type": "SiliconFlow",
    #     "model": "deepseek-ai/DeepSeek-V3",  # or any other model from siliconflow_model_cost.py
    #     "siliconflow_key": "your-api-key",
    #     "temperature": 0.7,
    #     "max_tokens": 150,
    #     "output_response": True
    # }
    
    ## _____________ Create a test agent _____________
    agent_payload = {
        "name": f"TestAgent_{uuid.uuid4().hex[:8]}",
        "description": "A test agent for execution simulation",
        "config": config,
        "runtime_params": {},
        "tags": ["test", "agent_generation"],
    }
    
    history = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "What about Germany?"},
        {"role": "assistant", "content": "The capital of Germany is Berlin."}
    ]
    
    
    ### ________ Testing query agent ________ ###
    agent_id = await create_agent(client, headers, agent_payload)
    
    # Query the agent through the API
    response = await client.post(
        urljoin(BASE_URL, f"agents/{agent_id}/query"),
        headers=headers,
        json={"prompt": "What is the capital of France?", "history": history}, 
        # timeout=20.0
    )
    print(response.json())
    result = response.json()
    assert response.status_code == 200
    assert "response" in result
    assert "Paris" in result["response"]

        
    # # Clean up
    # await delete_agent(client, headers, agent_id)
    assert False

    
# @pytest.mark.asyncio
# async def test_query_agent(client: httpx.AsyncClient, access_token: str):
#     headers = {"Authorization": f"Bearer {access_token}"}

    # # Create an Agent object directly
    # agent = Agent(
    #     name=f"TestAgent_{uuid.uuid4().hex[:8]}",
    #     description="A test agent for execution simulation",
    #     llm_config=llm_config,
    #     system_prompt="This is a system prompt for the agent.",
    #     use_long_term_memory=False
    # )
    
    
    # # Create an OpenAILLMConfig object
    # llm_config = OpenAILLMConfig(
    #     llm_type="OpenAILLM",
    #     model="gpt-3.5-turbo",
    #     openai_key=OPENAI_API_KEY,
    #     temperature=0.7,
    #     max_tokens=150,
    #     top_p=0.9,
    #     output_response=True
    # )
    
    # # Create an Agent object directly
    # agent = Agent(
    #     name=f"TestAgent_{uuid.uuid4().hex[:8]}",
    #     description="A test agent for execution simulation",
    #     llm_config=llm_config,
    #     system_prompt="This is a system prompt for the agent.",
    #     use_long_term_memory=False
    # )

    
    
    # ## _____________ Create a test agent _____________
    # agent_payload = {
    #     "name": f"TestAgent_{uuid.uuid4().hex[:8]}",
    #     "description": "A test agent for execution simulation",
    #     "config": {
    #         "llm_type":"OpenAILLM",
    #         "model":"gpt-3.5-turbo",
    #         "openai_key":OPENAI_API_KEY,
    #         "temperature":0.7,
    #         "max_tokens":150,
    #         "top_p":0.9,
    #         "output_response":True,
    #         "prompt": "You are a helpful assistant that can help with a variety of tasks.",
    #     },
    #     "runtime_params": {},
    #     "tags": ["test", "agent_generation"],
    # }
    
    # history = [
    #     {"role": "user", "content": "What is the capital of France?"},
    #     {"role": "assistant", "content": "The capital of France is Paris."},
    #     {"role": "user", "content": "What about Germany?"},
    #     {"role": "assistant", "content": "The capital of Germany is Berlin."}
    # ]
    
    # ### ___________ Agent Manager ___________ ###
    # agent_manager = AgentManager()
    # agent_manager.init_module()
    # agent_manager.add_agent({
    #     "name": agent_payload["name"],
    #     "description": agent_payload["description"],
    #     "prompt": "You are a helpful assistant that can help with a variety of tasks.",
    #     # "prompt": BOLT_PROMPT,
    #     "llm_config": llm_config,
    #     "config": agent_payload["config"],
    #     "runtime_params": agent_payload["runtime_params"],
    # })
    
    # agent = agent_manager.get_agent(agent_payload["name"])
    
    # print(agent.dict())
    
    #  # Retrieve the LLM configuration from the agent
    # llm_config_from_agent = agent.llm_config
    
    # # Initialize the OpenAILLM object using the retrieved configuration
    # openai_llm = OpenAILLM(config=llm_config_from_agent)
    # openai_llm.init_model()  # Initialize the model
    
    # print(agent.llm_config)
    
    # # Now you can use the openai_llm object to generate text or perform other actions
    # # For example, you can formulate a prompt and generate a response
    # prompt = "What is the capital of France?"
    # response = openai_llm.generate(
    #     prompt="And what about Italy?",
    #     system_message=agent.dict()["system_prompt"],
    #     history=history
    # )
    # print(response)
    
    
    # # # ___________ Batch Message _____________
    # # messages = openai_llm.formulate_messages(prompts=[prompt], 
    # #                                          system_messages=[agent.dict()["system_prompt"]])
    # # print("--------------------------------")
    # # print("--------------------------------")
    # # print("--------------------------------")
    # # print(messages)
    # # print("--------------------------------")
    # # print("--------------------------------")
    # # print("--------------------------------")
    # # response = openai_llm.single_generate(messages=messages[0])
    # # print("Generated Response:", response)

    # # assert False

