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

# Base API URL configuration
BASE_URL = "http://localhost:8000/api/v1/"
# OPENAI_API_KEY="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"
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


# @pytest.mark.asyncio
# async def test_successful_login(client: httpx.AsyncClient):
#     response = await client.post(
#         urljoin(BASE_URL, "auth/login"),
#         data={"username": "admin@clayx.ai", "password": "adminpassword"}
#     )
#     assert response.status_code == 200
#     json_resp = response.json()
#     assert "access_token" in json_resp

# @pytest.mark.asyncio
# async def test_failed_login(client: httpx.AsyncClient):
#     response = await client.post(
#         urljoin(BASE_URL, "auth/login"),
#         data={"username": "admin@clayx.ai", "password": "wrongpassword"}
#     )
#     assert response.status_code == 401

# # --- Test Cases --- #
# @pytest.mark.asyncio

# async def test_create_agent(client: httpx.AsyncClient, access_token: str):
#     headers = {"Authorization": f"Bearer {access_token}"}
#     payload = {
#         "name": f"TestAgent_{uuid.uuid4().hex[:8]}",
#         "description": "A test agent for integration testing",
#         "config": {"model": "gpt-3.5-turbo", "provider": "openai", "output_response": True},
#         "runtime_params": {},
#         "tags": ["test", "integration"]
#     }

#     agent_id = await create_agent(client, headers, payload)

#     # Debugging: Print API response
#     print(f"Agent ID: {agent_id}")

#     assert agent_id, "Agent ID should not be None or empty"

#     # Clean up
#     await delete_agent(client, headers, agent_id)

@pytest.mark.asyncio
async def test_execute_agent(client: httpx.AsyncClient, access_token: str):
    headers = {"Authorization": f"Bearer {access_token}"}
    
    
    # Create an OpenAILLMConfig object
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
    
    
    ## Create a test agent
    agent_payload = {
        "name": f"TestAgent_{uuid.uuid4().hex[:8]}",
        "description": "A test agent for execution simulation",
        "config": {
            "llm_type":"OpenAILLM",
            "model":"gpt-3.5-turbo",
            "openai_key":OPENAI_API_KEY,
            "temperature":0.7,
            "max_tokens":150,
            "top_p":0.9,
            "output_response":True,
        },
        "runtime_params": {},
        "tags": ["test", "execution"]
    }
    agent_id = await create_agent(client, headers, agent_payload)

    # Simulate executing the agent
    query_payload = {"query": "example task", "agent_id": agent_id}
    print(f"Agent ID: {agent_id}")
    print(f"Query Payload: {query_payload}")
    response = await client.post(
        urljoin(BASE_URL, f"agents/{agent_id}/execute"),
        headers=headers,
        json=query_payload
    )
    
    # Debugging: Print API response
    print(f"Execution Response: {response}")
    print(f"Execution Response: {response.json()}")

    # assert response.status_code == 200, f"Unexpected status code: {response.status_code}"
    # execution_data = response.json()
    # assert execution_data["_id"] == agent_id, "Returned agent ID should match the created agent ID"
    # assert execution_data["name"] == agent_payload["name"], "Returned agent name should match the created agent name"

    # # Print the agent information retrieved
    # print(f"Agent Information: {execution_data}")
    
    
    
    
    
    # Create an OpenAILLMConfig object
    
    # Create an Agent object directly
    # agent = Agent(
    #     name=response.json()[0],
    #     description=response.json()[1],
    #     llm_config=response.json()[2],
    #     agent_id=response.json()[3],
    #     use_long_term_memory=False
    # )
    
    # print(agent.dict())
    

    # Clean up
    await delete_agent(client, headers, agent_id)
    assert response.status_code == 300, f"Unexpected status code: {response.status_code}"

