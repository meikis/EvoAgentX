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
from evoagentx.prompts.requirement_collection import REQUIREMENT_COLLECTION_PROMPT


# Base API URL configuration
BASE_URL = "http://localhost:8000/api/v1/"
# OPENAI_API_KEY="sample"
OPENAI_API_KEY="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"

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
# async def test_execute_agent(client: httpx.AsyncClient, access_token: str):
#     headers = {"Authorization": f"Bearer {access_token}"}
#     #### ___________ Configurations ___________ ####
#     # OpenAI config
#     config = {
#         "llm_type": "OpenAILLM",
#         "model": "gpt-3.5-turbo",
#         "openai_key": OPENAI_API_KEY,
#         "temperature": 0.7,
#         "max_tokens": 150,
#         "top_p": 0.9,
#         "output_response": True,
#         "prompt": "You are a helpful assistant that can help with a variety of tasks.",
#     }

    
#     # # For OpenAI through LiteLLM
#     # config = {
#     #     "llm_type": "LiteLLM",
#     #     "model": "gpt-3.5-turbo",
#     #     "openai_key": "your-openai-key",
#     #     "temperature": 0.7,
#     #     "output_response": True
#     # }

#     # # For Anthropic through LiteLLM
#     # config = {
#     #     "llm_type": "LiteLLM",
#     #     "model": "anthropic/claude-2",
#     #     "anthropic_key": "your-anthropic-key",
#     #     "temperature": 0.7,
#     #     "output_response": True
#     # }

#     # # For DeepSeek through LiteLLM
#     # config = {
#     #     "llm_type": "LiteLLM",
#     #     "model": "deepseek-ai/deepseek-chat",
#     #     "deepseek_key": "your-deepseek-key",
#     #     "temperature": 0.7,
#     #     "output_response": True
#     # }
    
#     # config = {
#     #     "llm_type": "SiliconFlow",
#     #     "model": "deepseek-ai/DeepSeek-V3",  # or any other model from siliconflow_model_cost.py
#     #     "siliconflow_key": "your-api-key",
#     #     "temperature": 0.7,
#     #     "max_tokens": 150,
#     #     "output_response": True
#     # }
    
#     ## _____________ Create a test agent _____________
#     agent_payload = {
#         "name": f"TestAgent_{uuid.uuid4().hex[:8]}",
#         "description": "A test agent for execution simulation",
#         "config": config,
#         "runtime_params": {},
#         "tags": ["test", "agent_generation"],
#     }
    
#     history = [
#         {"role": "user", "content": "What is the capital of France?"},
#         {"role": "assistant", "content": "The capital of France is Paris."},
#         {"role": "user", "content": "What about Germany?"},
#         {"role": "assistant", "content": "The capital of Germany is Berlin."}
#     ]
    
    
#     ### ________ Testing query agent ________ ###
#     agent_id = await create_agent(client, headers, agent_payload)
    
#     # Query the agent through the API
#     response = await client.post(
#         urljoin(BASE_URL, f"agents/{agent_id}/query"),
#         headers=headers,
#         json={"prompt": "What is the capital of France?", "history": history}, 
#         # timeout=20.0
#     )
#     print(response.json())
#     result = response.json()
#     assert response.status_code == 200
#     assert "response" in result
#     assert "Paris" in result["response"]

# @pytest.mark.asyncio
# async def test_update_agent(client: httpx.AsyncClient, access_token: str):
#     headers = {"Authorization": f"Bearer {access_token}"}
    
#     # OpenAI config
#     config = {
#         "llm_type": "OpenAILLM",
#         "model": "gpt-3.5-turbo",
#         "openai_key": OPENAI_API_KEY,
#         "temperature": 0.7,
#         "max_tokens": 150,
#         "top_p": 0.9,
#         "output_response": True,
#         "prompt": "You are a helpful assistant that can help with a variety of tasks.",
#     }
    
#     # Create initial agent
#     agent_payload = {
#         "name": f"TestAgent_{uuid.uuid4().hex[:8]}",
#         "description": "Initial description",
#         "config": config,
#         "runtime_params": {},
#         "tags": ["test", "initial"],
#     }
    
#     agent_id = await create_agent(client, headers, agent_payload)
    
#     # Get initial agent details
#     initial_response = await client.get(
#         urljoin(BASE_URL, f"agents/{agent_id}"),
#         headers=headers
#     )
#     assert initial_response.status_code == 200
#     initial_agent = initial_response.json()
    
#     # Update agent data
#     update_payload = {
#         "name": f"UpdatedAgent_{uuid.uuid4().hex[:8]}",
#         "description": "Updated description",
#         "tags": ["test", "updated"],
#         "config": {
#             **config,
#             "temperature": 0.8,
#             "prompt": "You are an updated assistant with new capabilities."
#         }
#     }
    
#     # Send update request
#     response = await client.put(
#         urljoin(BASE_URL, f"agents/{agent_id}"),
#         headers=headers,
#         json=update_payload
#     )
    
#     # Verify response
#     assert response.status_code == 200
#     updated_agent = response.json()
#     print(updated_agent)
    
#     # Verify updated fields in database response
#     assert updated_agent["name"] == update_payload["name"]
#     assert updated_agent["description"] == update_payload["description"]
#     assert updated_agent["tags"] == update_payload["tags"]
#     assert updated_agent["config"]["temperature"] == update_payload["config"]["temperature"]
#     assert updated_agent["config"]["prompt"] == update_payload["config"]["prompt"]
    
#     # Verify agent ID is preserved
#     assert updated_agent["_id"] == initial_agent["_id"]
    
#     # Clean up
#     await delete_agent(client, headers, agent_id)
#     assert False

# @pytest.mark.asyncio
# async def test_agent_backup(client: httpx.AsyncClient, access_token: str):
#     headers = {"Authorization": f"Bearer {access_token}"}
    
#     # OpenAI config
#     config = {
#         "llm_type": "OpenAILLM",
#         "model": "gpt-3.5-turbo",
#         "openai_key": OPENAI_API_KEY,
#         "temperature": 0.7,
#         "max_tokens": 150,
#         "top_p": 0.9,
#         "output_response": True,
#         "prompt": "You are a helpful assistant that can help with a variety of tasks.",
#     }
    
#     # Create initial agent
#     agent_payload = {
#         "name": f"TestAgent_{uuid.uuid4().hex[:8]}",
#         "description": "Initial description",
#         "config": config,
#         "runtime_params": {},
#         "tags": ["test", "backup"],
#     }
    
#     agent_id = await create_agent(client, headers, agent_payload)
    
#     # Create backup directory
#     backup_dir = "backups"
#     if not os.path.exists(backup_dir):
#         os.makedirs(backup_dir)
    
#     # Create backup
#     backup_path = os.path.join(backup_dir, f"{agent_payload['name']}_v1.json")
#     response = await client.post(
#         urljoin(BASE_URL, f"agents/{agent_id}/backup"),
#         headers=headers,
#         params={"backup_path": backup_path}
#     )
#     assert response.status_code == 200
#     backup_result = response.json()
#     print(backup_result)
#     assert backup_result["success"] == True
#     assert backup_result["agent_id"] == agent_id
#     assert backup_result["backup_path"] == backup_path
    
#     # # List backups
#     # response = await client.get(
#     #     urljoin(BASE_URL, f"agents/{agent_id}/backups"),
#     #     headers=headers,
#     #     params={"backup_dir": backup_dir}
#     # )
#     # assert response.status_code == 200
#     # backups = response.json()
#     # print(backups)
#     # assert len(backups) > 0
#     # assert any(backup["path"] == backup_path for backup in backups)
    
#     # Load agent from backup
#     # loaded_agent = Agent.from_file(path=backup_path)
#     # print(loaded_agent.dict())
    
#     # print(backup_path)
#     # response = await client.post(
#     #     urljoin(BASE_URL, f"agents/{agent_id}/load_backup"),
#     #     headers=headers,
#     #     params={"backup_path": backup_path}
#     # )
#     # print(response.json())
    
#     # Restore from backup
#     response = await client.post(
#         urljoin(BASE_URL, f"agents/restore"),
#         headers=headers,
#         params={"backup_path": backup_path}
#     )
#     print(response)
#     print(response.json())
#     print("--------------------------------")
#     assert response.status_code == 200
#     restore_result = response.json()
#     assert restore_result["success"] == True
#     # Don't check that agent_id matches original - this is a new agent with a different ID
#     # assert restore_result["agent_id"] == agent_id
#     assert restore_result["backup_path"] == backup_path
    
#     # Clean up both agents
#     original_agent_id = agent_id
#     restored_agent_id = restore_result["agent_id"]
#     print(f"Original agent ID: {original_agent_id}")
#     print(f"Restored agent ID: {restored_agent_id}")
    
#     # Delete the restored agent
#     await delete_agent(client, headers, restored_agent_id)
#     # Delete the original agent
#     await delete_agent(client, headers, original_agent_id)

# @pytest.mark.asyncio
# async def test_batch_agent_backup_restore(client: httpx.AsyncClient, access_token: str):
#     """Test the batch backup and restore functionality for agents."""
#     headers = {"Authorization": f"Bearer {access_token}"}
    
#     # Create a backup directory
#     backup_dir = "batch_backups"
#     if not os.path.exists(backup_dir):
#         os.makedirs(backup_dir)
    
#     # Create 3 test agents for batch operations
#     agent_ids = []
#     for i in range(3):
#         agent_payload = {
#             "name": f"BatchTestAgent_{i}_{uuid.uuid4().hex[:8]}",
#             "description": f"Test agent {i} for batch backup/restore",
#             "config": {
#                 "llm_type": "OpenAILLM",
#                 "model": "gpt-3.5-turbo",
#                 "openai_key": OPENAI_API_KEY,
#                 "temperature": 0.7,
#                 "max_tokens": 150,
#                 "top_p": 0.9,
#                 "output_response": True,
#                 "prompt": f"You are helpful assistant {i}."
#             },
#             "runtime_params": {},
#             "tags": ["test", "batch"]
#         }
#         agent_id = await create_agent(client, headers, agent_payload)
#         agent_ids.append(agent_id)
        
#     try:
#         # Test backing up multiple agents
#         response = await client.post(
#             urljoin(BASE_URL, "agents/backup-batch"),
#             headers=headers,
#             params={"backup_dir": backup_dir},
#             json=agent_ids
#         )
#         assert response.status_code == 200
#         backup_result = response.json()
#         print(f"Batch backup result: {backup_result}")
#         assert backup_result["success"] == True
#         assert backup_result["total"] == 3
#         assert backup_result["successful"] == 3
        
#         # Delete the original agents to test restoration
#         for agent_id in agent_ids:
#             await delete_agent(client, headers, agent_id)
        
#         # Find all backup files
#         backup_files = []
#         for filename in os.listdir(backup_dir):
#             if filename.startswith("BatchTestAgent_") and filename.endswith(".json"):
#                 backup_files.append(os.path.join(backup_dir, filename))
        
#         # Test restoring multiple agents from backups
#         response = await client.post(
#             urljoin(BASE_URL, "agents/restore-batch"),
#             headers=headers,
#             json=backup_files
#         )
#         assert response.status_code == 200
#         restore_result = response.json()
#         print(f"Batch restore result: {restore_result}")
#         assert restore_result["success"] == True
#         assert restore_result["total"] == len(backup_files)
#         assert restore_result["successful"] == len(backup_files)
        
#         # Get the IDs of the restored agents
#         restored_agent_ids = [result["agent_id"] for result in restore_result["results"]]
        
#         # Clean up - delete the restored agents
#         for agent_id in restored_agent_ids:
#             await delete_agent(client, headers, agent_id)
            
#     finally:
#         # Clean up any remaining agents
#         for agent_id in agent_ids:
#             try:
#                 await delete_agent(client, headers, agent_id)
#             except:
#                 pass
        
#         # Clean up backup directory
#         import shutil
#         if os.path.exists(backup_dir):
#             shutil.rmtree(backup_dir)

@pytest.mark.asyncio
async def test_backup_restore_all_agents(client: httpx.AsyncClient, access_token: str):
    """Test backing up and restoring all agents in the AgentManager."""
    headers = {"Authorization": f"Bearer {access_token}"}
    
    # Create backup directories
    all_backup_dir = "all_backups"
    if not os.path.exists(all_backup_dir):
        os.makedirs(all_backup_dir)
    
    # Create 2 more test agents
    additional_agent_ids = []
    for i in range(2):
        agent_payload = {
            "name": f"AllAgentTest_{i}_{uuid.uuid4().hex[:8]}",
            "description": f"Test agent {i} for all-agents backup/restore",
            "config": {
                "llm_type": "OpenAILLM",
                "model": "gpt-3.5-turbo",
                "openai_key": OPENAI_API_KEY,
                "temperature": 0.7,
                "max_tokens": 150,
                "top_p": 0.9,
                "output_response": True,
                "prompt": f"You are helpful assistant {i}."
            },
            "runtime_params": {},
            "tags": ["test", "all_agents"]
        }
        agent_id = await create_agent(client, headers, agent_payload)
        additional_agent_ids.append(agent_id)
    
    try:
        # Test backing up all agents
        response = await client.post(
            urljoin(BASE_URL, "agents/backup-all"),
            headers=headers,
            params={"backup_dir": all_backup_dir}
        )
        assert response.status_code == 200
        backup_result = response.json()
        print(f"Backup all agents result: {backup_result}")
        assert backup_result["success"] == True
        
        # Delete our test agents
        for agent_id in additional_agent_ids:
            await delete_agent(client, headers, agent_id)
        
        # Test restoring all agents from the backup directory
        response = await client.post(
            urljoin(BASE_URL, "agents/restore-all"),
            headers=headers,
            params={"backup_dir": all_backup_dir}
        )
        assert response.status_code == 200
        restore_result = response.json()
        print(f"Restore all agents result: {restore_result}")
        assert restore_result["success"] == True
        
        # Get the IDs of the restored agents - but only the ones we added
        restored_agents = []
        for result in restore_result["results"]:
            if "agent_name" in result and result["agent_name"].startswith("AllAgentTest_"):
                restored_agents.append(result)
        
        # We should have at least our 2 additional agents restored
        assert len(restored_agents) >= 2
        
        # Clean up - delete the restored agents
        for result in restored_agents:
            if "agent_id" in result:
                try:
                    await delete_agent(client, headers, result["agent_id"])
                except:
                    pass
    
    finally:
        # Clean up any remaining agents
        for agent_id in additional_agent_ids:
            try:
                await delete_agent(client, headers, agent_id)
            except:
                pass
        
        # Clean up backup directory
        import shutil
        if os.path.exists(all_backup_dir):
            shutil.rmtree(all_backup_dir)



    
# @pytest.mark.asyncio
# async def test_query_agent(client: httpx.AsyncClient, access_token: str):
#     headers = {"Authorization": f"Bearer {access_token}"}
#     # Create an OpenAILLMConfig object
#     config = OpenAILLMConfig(
#         llm_type="OpenAILLM",
#         model="gpt-3.5-turbo",
#         openai_key=OPENAI_API_KEY,
#         temperature=0.7,
#         max_tokens=150,
#         top_p=0.9,
#         output_response=True
#     )
#    # Create an Agent object directly
    
    
    
    
#     # # Create an Agent object directly
#     # agent = Agent(
#     #     name=f"TestAgent_{uuid.uuid4().hex[:8]}",
#     #     description="A test agent for execution simulation",
#     #     llm_config=config,
#     #     system_prompt="This is a system prompt for the agent.",
#     #     use_long_term_memory=False
#     # )

    
    
#     ## _____________ Create a test agent _____________
#     agent_payload = {
#         "name": f"TestAgent_{uuid.uuid4().hex[:8]}",
#         "description": "A test agent for execution simulation",
#         "config": config,
#         "runtime_params": {},
#         "tags": ["test", "agent_generation"],
#     }
    
#     history = [
#         {"role": "user", "content": "What is the capital of France?"},
#         {"role": "assistant", "content": "The capital of France is Paris."},
#         {"role": "user", "content": "What about Germany?"},
#         {"role": "assistant", "content": "The capital of Germany is Berlin."}
#     ]
    
#     ### ___________ Agent Manager ___________ ###
#     agent_manager = AgentManager()
#     agent_manager.init_module()
#     agent_manager.add_agent({
#         "name": agent_payload["name"],
#         "description": agent_payload["description"],
#         "prompt": REQUIREMENT_COLLECTION_PROMPT,
#         # "prompt": BOLT_PROMPT,
#         "llm_config": config,
#         "config": agent_payload["config"],
#         "runtime_params": agent_payload["runtime_params"],
#     })
    
#     agent = agent_manager.get_agent(agent_payload["name"])
    
#     print(agent.dict())
    
#      # Retrieve the LLM configuration from the agent
#     llm_config_from_agent = agent.llm_config
    
#     # Initialize the OpenAILLM object using the retrieved configuration
#     openai_llm = OpenAILLM(config=llm_config_from_agent)
#     openai_llm.init_model()  # Initialize the model
    
#     print(agent.llm_config)
    
#     # Now you can use the openai_llm object to generate text or perform other actions
#     # For example, you can formulate a prompt and generate a response
#     response = openai_llm.generate(
#         prompt="I am up to a project to build a website for a company. It is about a blog website that features articles, comments, and a search functionality. The website should be responsive and easy to navigate. The website should be secure and have a user authentication system. The website should be fast and have a good user experience. The website should be scalable and have a good user experience. The website should be scalable and have a good user experience. The website should be scalable and have a good user experience.",
#         system_message=agent.dict()["system_prompt"],
#         # history=history
#     )
#     print(response)
#     assert False
    
    
    # # ___________ Batch Message _____________
    # messages = openai_llm.formulate_messages(prompts=[prompt], 
    #                                          system_messages=[agent.dict()["system_prompt"]])
    # print("--------------------------------")
    # print("--------------------------------")
    # print("--------------------------------")
    # print(messages)
    # print("--------------------------------")
    # print("--------------------------------")
    # print("--------------------------------")
    # response = openai_llm.single_generate(messages=messages[0])
    # print("Generated Response:", response)

    # assert False

    
# @pytest.mark.asyncio
# async def test_execution(client: httpx.AsyncClient, access_token: str):
#     headers = {"Authorization": f"Bearer {access_token}"}
#     #### ___________ Configurations ___________ ####
#     # OpenAI config
    
#     config = {
#         "llm_type": "OpenAILLM",
#         "model": "gpt-3.5-turbo",
#         "openai_key": OPENAI_API_KEY,
#         "temperature": 0.7,
#         "max_tokens": 500,
#         "top_p": 0.9,
#         "output_response": True,
#         "stream": False,
#     }
    

#     ## _____________ Create a test agent _____________
#     agent_payload = {
#         "name": f"TestAgent_{uuid.uuid4().hex[:8]}",
#         "description": "A test agent for execution simulation",
#         "config": config,
#         "runtime_params": {},
#         "tags": ["test", "agent_generation"],
#     }
    
#     agent_manager = AgentManager()
#     agent_manager.init_module()
#     agent_manager.add_agent({
#         "name": agent_payload["name"],
#         "description": agent_payload["description"],
#         "prompt": "You are a helpful assistant that can help with a variety of tasks.",  # Changed back to prompt from system_prompt
#         "llm_config": config,
#         "config": agent_payload["config"],
#         "runtime_params": agent_payload["runtime_params"],
#     })
    
#     agent = agent_manager.get_agent(agent_payload["name"])
    
#     agent_generation_action = AgentGeneration(name="AgentGeneration")
#     agent.add_action(agent_generation_action)
    
#     # Execute the AgentGeneration action
#     action_input_data = {
#         "goal": "Create an agent to manage a project",
#         "workflow": "Project management workflow",
#         "task": '{"name": "Manage project", "description": "Manage project tasks", "inputs": [], "outputs": []}',
#         "history": None,
#         "suggestion": "Use agile methodology",
#         "existing_agents": None,
#         "tools": None
#     }
    
#     result = agent.execute(
#         action_name="AgentGeneration", 
#         action_input_data=action_input_data
#     )
#     print("Execution Result:", result.content)
    
#     assert "agent" in str(result.content).lower() or "project" in str(result.content).lower()
#     # assert False
#     # agent_id = await create_agent(client, headers, agent_payload)
#     # print(agent_id)
#     # await delete_agent(client, headers, agent_id)


#     # # Simulate executing the agent
#     # query_payload = {"query": "example task", "agent_id": agent_id}
#     # print(f"Agent ID: {agent_id}")
#     # print(f"Query Payload: {query_payload}")
#     # response = await client.post(
#     #     urljoin(BASE_URL, f"agents/{agent_id}/execute"),
#     #     headers=headers,
#     #     json=query_payload
#     # )
    
#     # # Debugging: Print API response
#     # print(f"Execution Response: {response}")
#     # print(f"Execution Response: {response.json()}")

#     # assert response.status_code == 200, f"Unexpected status code: {response.status_code}"
#     # execution_data = response.json()
#     # assert execution_data["_id"] == agent_id, "Returned agent ID should match the created agent ID"
#     # assert execution_data["name"] == agent_payload["name"], "Returned agent name should match the created agent name"

#     # # Print the agent information retrieved
#     # print(f"Agent Information: {execution_data}")