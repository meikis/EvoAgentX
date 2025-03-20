"""
Business logic for agents, workflows, and executions.
"""
import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Tuple
from bson import ObjectId
from evoagentx.models.openai_model import OpenAILLM
from evoagentx.models.model_configs import OpenAILLMConfig

from evoagentx.app.db import (
    Database, Agent, Workflow, WorkflowExecution, ExecutionLog,
    AgentStatus, WorkflowStatus, ExecutionStatus
)
from evoagentx.app.schemas import (
    AgentCreate, AgentUpdate, WorkflowCreate, WorkflowUpdate, 
    ExecutionCreate, PaginationParams, SearchParams, AgentQueryRequest
)
from evoagentx.app.shared import agent_manager


logger = logging.getLogger(__name__)

# Agent Service
class AgentService:
    @staticmethod
    async def create_agent(agent_data: AgentCreate, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Create a new agent and add it to the AgentManager."""
        agent_dict = agent_data.dict()
        agent_dict["created_by"] = user_id
        agent_dict["created_at"] = datetime.utcnow()
        agent_dict["updated_at"] = agent_dict["created_at"]
        agent_dict["status"] = AgentStatus.CREATED
        
        # Validate agent exists with the same name
        existing_agent = await Database.agents.find_one({"name": agent_dict["name"]})
        if existing_agent:
            raise ValueError(f"Agent with name '{agent_dict['name']}' already exists")
                
        try:
            # Create OpenAILLMConfig from the configuration
            llm_config = OpenAILLMConfig(
                llm_type=agent_dict["config"]["llm_type"],
                model=agent_dict["config"]["model"],
                openai_key=agent_dict["config"]["openai_key"],
                temperature=agent_dict["config"]["temperature"],
                max_tokens=agent_dict["config"]["max_tokens"],
                top_p=agent_dict["config"]["top_p"],
                output_response=agent_dict["config"]["output_response"]
            )
            
            agent_manager.add_agent({
                "name": agent_dict["name"],
                "description": agent_dict["description"],
                "prompt": agent_dict["config"]["prompt"],
                "llm_config": llm_config,
                "runtime_params": agent_dict["runtime_params"],
            })
            logger.info(f"Added agent {agent_dict['name']} to AgentManager")
        except Exception as e:
            logger.error(f"Failed to add agent {agent_dict['name']} to AgentManager: {e}, agent_dict: {agent_dict}")
            raise ValueError(f"Failed to add agent to AgentManager: {e}")
        
        try:
            result = await Database.agents.insert_one(agent_dict)
            agent_dict["_id"] = result.inserted_id
            logger.info(f"Created agent {agent_dict['name']} with ID {result.inserted_id}")
            return agent_dict
        except Exception as e:
            logger.error(f"Failed to add agent {agent_dict['name']} to Database: {e}, agent_dict: {agent_dict}")
            raise ValueError(f"Failed to add agent to Database: {e}")
        

    
    @staticmethod
    async def get_agent(agent_id: str) -> Optional[Dict[str, Any]]:
        """Get an agent by ID."""
        if not ObjectId.is_valid(agent_id):
            raise ValueError(f"Invalid agent ID: {agent_id}")
            
        agent = await Database.agents.find_one({"_id": ObjectId(agent_id)})
        return agent
    
    @staticmethod
    async def get_agent_by_name(name: str) -> Optional[Dict[str, Any]]:
        """Get an agent by name."""
        return await Database.agents.find_one({"name": name})
    
    @staticmethod
    async def update_agent(agent_id: str, agent_data: AgentUpdate) -> Optional[Dict[str, Any]]:
        """Update an agent."""
        if not ObjectId.is_valid(agent_id):
            raise ValueError(f"Invalid agent ID: {agent_id}")
            
        agent = await Database.agents.find_one({"_id": ObjectId(agent_id)})
        if not agent:
            return None
        
        update_data = agent_data.dict(exclude_unset=True)
        update_data["updated_at"] = datetime.utcnow()
        
        if "name" in update_data:
            # Check if the new name already exists
            existing = await Database.agents.find_one({
                "name": update_data["name"],
                "_id": {"$ne": ObjectId(agent_id)}
            })
            if existing:
                raise ValueError(f"Agent with name '{update_data['name']}' already exists")
        
        # Handle partial config updates
        if "config" in update_data and isinstance(update_data["config"], dict) and isinstance(agent.get("config"), dict):
            # Instead of replacing the entire config, merge the updated fields with existing ones
            merged_config = agent["config"].copy()
            for key, value in update_data["config"].items():
                merged_config[key] = value
            update_data["config"] = merged_config
            logger.info(f"Merged config update for agent {agent_id}: {update_data['config']}")
        
        # Update in database
        await Database.agents.update_one(
            {"_id": ObjectId(agent_id)},
            {"$set": update_data}
        )
        
        # Get the updated agent with all fields merged
        updated_agent = await Database.agents.find_one({"_id": ObjectId(agent_id)})
        
        # Update in agent_manager if it exists there
        agent_id_to_preserve = None
        try:
            # Check if agent exists in manager
            agent_exists_in_manager = agent_manager.has_agent(agent_name=agent["name"])
            
            if agent_exists_in_manager:
                # Get the current agent from manager to preserve its ID
                current_agent = agent_manager.get_agent(agent_name=agent["name"])
                agent_id_to_preserve = current_agent.agent_id
                
                # Remove old agent from manager
                agent_manager.remove_agent(agent_name=agent["name"])
            
            # Create config
            llm_config = OpenAILLMConfig(
                llm_type=updated_agent["config"]["llm_type"],
                model=updated_agent["config"]["model"],
                openai_key=updated_agent["config"]["openai_key"],
                temperature=updated_agent["config"]["temperature"],
                max_tokens=updated_agent["config"]["max_tokens"],
                top_p=updated_agent["config"]["top_p"],
                output_response=updated_agent["config"]["output_response"]
            )
            
            # Add updated agent to manager with preserved ID if it existed before
            agent_dict = {
                "name": updated_agent["name"],
                "description": updated_agent["description"],
                "prompt": updated_agent["config"]["prompt"],
                "llm_config": llm_config,
                "runtime_params": updated_agent["runtime_params"]
            }
            
            # Add the agent_id only if we had one before
            if agent_id_to_preserve:
                agent_dict["agent_id"] = agent_id_to_preserve
                
            agent_manager.add_agent(agent_dict)
            logger.info(f"Updated agent {agent_id} in AgentManager")
        except Exception as e:
            logger.error(f"Failed to update agent {agent_id} in AgentManager: {e}")
            # Don't raise an error here - we want the database update to succeed
            # even if the agent_manager update fails
        
        logger.info(f"Updated agent {agent_id}")
        
        return updated_agent
    
    @staticmethod
    async def delete_agent(agent_id: str) -> bool:
        """Delete an agent and remove it from the AgentManager."""
        if not ObjectId.is_valid(agent_id):
            raise ValueError(f"Invalid agent ID: {agent_id}")
          
        # Remove the agent from the AgentManager
        agent = await Database.agents.find_one({"_id": ObjectId(agent_id)})
        if agent:
            agent_manager.remove_agent(agent_name=agent["name"])
          
        # Check if agent is used in any workflows
        workflow_count = await Database.workflows.count_documents({"agent_ids": agent_id})
        if workflow_count > 0:
            raise ValueError(f"Cannot delete agent {agent_id} as it is used in {workflow_count} workflows")

        result = await Database.agents.delete_one({"_id": ObjectId(agent_id)})
        if result.deleted_count:
            logger.info(f"Deleted agent {agent_id}")
            return True
        return False
    
    @staticmethod
    async def list_agents(
        params: PaginationParams, 
        search: Optional[SearchParams] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """List agents with pagination and search."""
        query = {}
        
        if search:
            if search.query:
                query["$text"] = {"$search": search.query}
            
            if search.tags:
                query["tags"] = {"$all": search.tags}
            
            if search.status:
                query["status"] = search.status
            
            if search.start_date and search.end_date:
                query["created_at"] = {
                    "$gte": search.start_date,
                    "$lte": search.end_date
                }
            elif search.start_date:
                query["created_at"] = {"$gte": search.start_date}
            elif search.end_date:
                query["created_at"] = {"$lte": search.end_date}
        
        total = await Database.agents.count_documents(query)
        
        cursor = Database.agents.find(query)\
            .sort("created_at", -1)\
            .skip(params.skip)\
            .limit(params.limit)
        
        agents = await cursor.to_list(length=params.limit)
        return agents, total
    
    @staticmethod
    async def query_agent(agent_id: str, query: AgentQueryRequest, user_id: Optional[str] = None) -> str:
        """Query an agent with a prompt and get a response."""
        # Get the agent from database
        agent_data = await Database.agents.find_one({"_id": ObjectId(agent_id)})
        if not agent_data:
            raise ValueError(f"Agent with ID {agent_id} not found")

        try:
            # Get the agent instance from the manager
            agent_instance = agent_manager.get_agent(agent_name=agent_data["name"])
            
            # Get the model type from config
            llm_type = agent_instance.llm_config.llm_type.lower()
            
            # Initialize the appropriate LLM based on type
            if llm_type == "openaillm":
                from evoagentx.models.openai_model import OpenAILLM
                llm = OpenAILLM(config=agent_instance.llm_config)
            elif llm_type == "litellm":
                from evoagentx.models.litellm_model import LiteLLM
                llm = LiteLLM(config=agent_instance.llm_config)
            elif llm_type == "siliconflow":
                from evoagentx.models.siliconflow_model import SiliconFlowLLM
                llm = SiliconFlowLLM(config=agent_instance.llm_config)
            else:
                raise ValueError(f"Unsupported model type: {llm_type}")
            
            # Initialize the model
            llm.init_model()
            
            # Format messages with history
            messages = []
            
            # Add system message if present
            if agent_instance.system_prompt:
                messages.append({"role": "system", "content": agent_instance.system_prompt})
            
            # Add conversation history if provided
            if query.history:
                messages.extend(query.history)
            # Add the current user message
            messages.append({"role": "user", "content": query.prompt})
  
            # Generate response using single_generate
            response = llm.single_generate(messages=messages)
            return response
            
        except Exception as e:
            logger.error(f"Error querying agent {agent_id}: {str(e)}")
            raise ValueError(f"Error querying agent: {str(e)}")

    # @staticmethod
    # async def execute_agent_action(agent_id: str, action_name: str, action_params: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    #     """Execute an action using a specific agent."""
    #     try:
    #         # Validate agent ID
    #         agent = await Database.agents.find_one({"_id": ObjectId(agent_id)})
    #         if not agent:
    #             return {"success": False, "error": "Agent not found"}

    #         # Check if the current user is the creator of the agent
    #         # if agent["created_by"] != user_id:
    #         #     return {"success": False, "error": "You do not have permission to execute this agent"}

    #         # Retrieve the agent instance from the manager
    #         agent_instance = agent_manager.get_agent(agent_name=agent["name"])

    #         # Check if the action is valid
    #         if action_name not in [action.name for action in agent_instance.get_all_actions()]:
    #             return {"success": False, "error": f"Invalid action '{action_name}' for agent '{agent['name']}', allowed actions: {agent_instance.get_all_actions()}"}

    #         # Execute the action
    #         result_message = agent_instance.execute(
    #             action_name=action_name,
    #             action_input_data=action_params
    #         )

    #         # Return the result
    #         return {
    #             "success": True,
    #             "result": result_message.content,
    #             "prompt": result_message.prompt
    #         }

    #     except Exception as e:
    #         logger.error(f"Error executing action '{action_name}' for agent '{agent_id}': {str(e)}")
    #         return {"success": False, "error": f"Internal server error: {str(e)}"}

# Workflow Service
class WorkflowService:
    @staticmethod
    async def create_workflow(workflow_data: WorkflowCreate, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Create a new workflow."""
        workflow_dict = workflow_data.dict()
        workflow_dict["created_by"] = user_id
        workflow_dict["created_at"] = datetime.utcnow()
        workflow_dict["updated_at"] = workflow_dict["created_at"]
        workflow_dict["status"] = WorkflowStatus.CREATED
        workflow_dict["version"] = 1
        
        # Extract agent IDs from the workflow definition
        agent_ids = set()
        
        # Extract agent IDs from steps
        steps = workflow_dict["definition"].get("steps", [])
        for step in steps:
            if "agent_id" in step:
                agent_id = step["agent_id"]
                # Validate agent exists
                agent = await AgentService.get_agent(agent_id)
                if not agent:
                    raise ValueError(f"Agent with ID {agent_id} does not exist")
                agent_ids.add(agent_id)
        
        workflow_dict["agent_ids"] = list(agent_ids)
        
        # Check for existing workflow with the same name
        existing = await Database.workflows.find_one({"name": workflow_dict["name"]})
        if existing:
            raise ValueError(f"Workflow with name '{workflow_dict['name']}' already exists")
        
        result = await Database.workflows.insert_one(workflow_dict)
        workflow_dict["_id"] = result.inserted_id
        
        logger.info(f"Created workflow {workflow_dict['name']} with ID {result.inserted_id}")
        
        return workflow_dict
    
    @staticmethod
    async def get_workflow(workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get a workflow by ID."""
        if not ObjectId.is_valid(workflow_id):
            raise ValueError(f"Invalid workflow ID: {workflow_id}")
        workflow = await Database.workflows.find_one({"_id": ObjectId(workflow_id)})
        return workflow
    
    @staticmethod
    async def get_workflow_by_name(name: str) -> Optional[Dict[str, Any]]:
        """Get a workflow by name."""
        return await Database.workflows.find_one({"name": name})
    
    @staticmethod
    async def update_workflow(workflow_id: str, workflow_data: WorkflowUpdate) -> Optional[Dict[str, Any]]:
        """Update a workflow."""
        if not ObjectId.is_valid(workflow_id):
            raise ValueError(f"Invalid workflow ID: {workflow_id}")
            
        workflow = await Database.workflows.find_one({"_id": ObjectId(workflow_id)})
        if not workflow:
            return None
        
        update_data = workflow_data.dict(exclude_unset=True)
        update_data["updated_at"] = datetime.utcnow()
        
        # Update version if definition changes
        if "definition" in update_data:
            update_data["version"] = workflow.get("version", 1) + 1
            
            # Extract agent IDs from the updated workflow definition
            agent_ids = set()
            steps = update_data["definition"].get("steps", [])
            for step in steps:
                if "agent_id" in step:
                    agent_id = step["agent_id"]
                    # Validate agent exists
                    agent = await AgentService.get_agent(agent_id)
                    if not agent:
                        raise ValueError(f"Agent with ID {agent_id} does not exist")
                    agent_ids.add(agent_id)
            
            update_data["agent_ids"] = list(agent_ids)
        
        # Check for name conflict if name is being updated
        if "name" in update_data:
            existing = await Database.workflows.find_one({
                "name": update_data["name"],
                "_id": {"$ne": ObjectId(workflow_id)}
            })
            if existing:
                raise ValueError(f"Workflow with name '{update_data['name']}' already exists")
        
        await Database.workflows.update_one(
            {"_id": ObjectId(workflow_id)},
            {"$set": update_data}
        )
        
        updated_workflow = await Database.workflows.find_one({"_id": ObjectId(workflow_id)})
        logger.info(f"Updated workflow {workflow_id}")
        
        return updated_workflow
    
    @staticmethod
    async def delete_workflow(workflow_id: str) -> bool:
        """Delete a workflow."""
        if not ObjectId.is_valid(workflow_id):
            raise ValueError(f"Invalid workflow ID: {workflow_id}")
        
        # Find all executions for the workflow
        executions = await Database.executions.find({"workflow_id": workflow_id}).to_list(length=None)
        
        # Separate pending and running executions
        pending_executions = [exec for exec in executions if exec["status"] == ExecutionStatus.PENDING]
        running_executions = [exec for exec in executions if exec["status"] == ExecutionStatus.RUNNING]
        
        # If there are running executions, raise an error
        if running_executions:
            raise ValueError(f"Cannot delete workflow {workflow_id} with {len(running_executions)} running executions")
        
        # Stop and remove pending executions
        for exec in pending_executions:
            await WorkflowExecutionService.update_execution_status(
                execution_id=str(exec["_id"]),
                status=ExecutionStatus.CANCELLED
            )
            await Database.executions.delete_one({"_id": exec["_id"]})
            logger.info(f"Stopped and removed pending execution {exec['_id']} for workflow {workflow_id}")
        
        # Delete the workflow
        result = await Database.workflows.delete_one({"_id": ObjectId(workflow_id)})
        if result.deleted_count:
            # Optionally, delete associated logs
            await Database.logs.delete_many({"workflow_id": workflow_id})
            logger.info(f"Deleted workflow {workflow_id}")
            return True
        
        return False
    
    @staticmethod
    async def list_workflows(
        params: PaginationParams, 
        search: Optional[SearchParams] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """List workflows with pagination and search."""
        query = {}
        
        if search:
            if search.query:
                query["$text"] = {"$search": search.query}
            
            if search.tags:
                query["tags"] = {"$all": search.tags}
            
            if search.status:
                query["status"] = search.status
            
            if search.start_date and search.end_date:
                query["created_at"] = {
                    "$gte": search.start_date,
                    "$lte": search.end_date
                }
            elif search.start_date:
                query["created_at"] = {"$gte": search.start_date}
            elif search.end_date:
                query["created_at"] = {"$lte": search.end_date}
        
        total = await Database.workflows.count_documents(query)
        
        cursor = Database.workflows.find(query)\
            .sort("created_at", -1)\
            .skip(params.skip)\
            .limit(params.limit)
        
        workflows = await cursor.to_list(length=params.limit)
        return workflows, total

# Workflow Execution Service
class WorkflowExecutionService:
    @staticmethod
    async def create_execution(execution_data: ExecutionCreate, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Create a new workflow execution."""
        # Validate workflow exists
        workflow = await WorkflowService.get_workflow(execution_data.workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {execution_data.workflow_id} not found")
        
        # Prepare execution document
        execution_dict = {
            "workflow_id": execution_data.workflow_id,
            "status": ExecutionStatus.PENDING,
            "start_time": datetime.utcnow(),
            "input_params": execution_data.input_params,
            "created_by": user_id,
            "created_at": datetime.utcnow(),
            "step_results": {},
            "current_step": None,
            "results": {},
            "error_message": None
        }
        
        # Insert execution record
        result = await Database.executions.insert_one(execution_dict)
        execution_dict["_id"] = result.inserted_id
        
        logger.info(f"Created workflow execution {result.inserted_id}")
        
        # Optional: Queue execution for async processing
        # This would typically use a task queue like Celery
        # await execute_workflow_async.delay(execution_dict)
        
        return execution_dict
    
    @staticmethod
    async def get_execution(execution_id: str) -> Optional[Dict[str, Any]]:
        """Get a workflow execution by ID."""
        if not ObjectId.is_valid(execution_id):
            raise ValueError(f"Invalid execution ID: {execution_id}")
            
        execution = await Database.executions.find_one({"_id": ObjectId(execution_id)})
        return execution
    
    @staticmethod
    async def update_execution_status(execution_id: str, status: ExecutionStatus, error_message: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Update execution status."""
        if not ObjectId.is_valid(execution_id):
            raise ValueError(f"Invalid execution ID: {execution_id}")
        
        update_data = {
            "status": status,
            "updated_at": datetime.utcnow()
        }
        
        if status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED]:
            update_data["end_time"] = datetime.utcnow()
        
        if error_message:
            update_data["error_message"] = error_message
        
        result = await Database.executions.find_one_and_update(
            {"_id": ObjectId(execution_id)},
            {"$set": update_data},
            return_document=True
        )
        
        return result
    
    @staticmethod
    async def list_executions(
        workflow_id: Optional[str] = None,
        params: PaginationParams = PaginationParams(), 
        search: Optional[SearchParams] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """List workflow executions with pagination and search."""
        query = {}
        
        if workflow_id:
            query["workflow_id"] = workflow_id
        
        if search:
            if search.status:
                query["status"] = search.status
            
            if search.start_date and search.end_date:
                query["created_at"] = {
                    "$gte": search.start_date,
                    "$lte": search.end_date
                }
            elif search.start_date:
                query["created_at"] = {"$gte": search.start_date}
            elif search.end_date:
                query["created_at"] = {"$lte": search.end_date}
        
        total = await Database.executions.count_documents(query)
        
        cursor = Database.executions.find(query)\
            .sort("created_at", -1)\
            .skip(params.skip)\
            .limit(params.limit)
        
        executions = await cursor.to_list(length=params.limit)
        return executions, total
    
    @staticmethod
    async def log_execution_event(
        workflow_id: str, 
        execution_id: str, 
        message: str,
        step_id: Optional[str] = None, 
        agent_id: Optional[str] = None, 
        level: str = "INFO", 
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Log an event in a workflow execution."""
        log_entry = {
            "workflow_id": workflow_id,
            "execution_id": execution_id,
            "step_id": step_id,
            "agent_id": agent_id,
            "timestamp": datetime.utcnow(),
            "level": level,
            "message": message,
            "details": details or {}
        }
        
        result = await Database.logs.insert_one(log_entry)
        log_entry["_id"] = result.inserted_id
        
        return log_entry
    
    @staticmethod
    async def get_execution_logs(
        execution_id: str, 
        params: PaginationParams = PaginationParams()
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Retrieve logs for a specific execution."""
        query = {"execution_id": execution_id}
        
        total = await Database.logs.count_documents(query)
        
        cursor = Database.logs.find(query)\
            .sort("timestamp", 1)\
            .skip(params.skip)\
            .limit(params.limit)
        
        logs = await cursor.to_list(length=params.limit)
        return logs, total

class AgentBackupService:
    @staticmethod
    async def save_agent_backup(agent_id: str, backup_path: str) -> Dict[str, Any]:
        """Save an agent's current state to a backup file."""
        if not ObjectId.is_valid(agent_id):
            raise ValueError(f"Invalid agent ID: {agent_id}")
            
        # Get the agent from database
        agent = await Database.agents.find_one({"_id": ObjectId(agent_id)})
        if not agent:
            raise ValueError(f"Agent with ID {agent_id} not found in database")
            
        # Check if agent exists in the AgentManager
        agent_name = agent["name"]
        if not agent_manager.has_agent(agent_name):
            raise ValueError(f"Agent '{agent_name}' is not in the AgentManager (not running)")
            
        # Get the agent instance from manager
        agent_instance = agent_manager.get_agent(agent_name=agent["name"])
        
        # Create backup directory if it doesn't exist
        import os
        backup_dir = os.path.dirname(backup_path)
        if backup_dir and not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
        
        # Save the agent to the specified path
        try:
            agent_instance.save_module(path=backup_path)
            return {
                "success": True,
                "message": f"Agent backup saved to {backup_path}",
                "agent_id": agent_id,
                "backup_path": backup_path,
                "agent_name": agent["name"]
            }
        except Exception as e:
            logger.error(f"Failed to save agent backup: {str(e)}")
            raise ValueError(f"Failed to save agent backup: {str(e)}")
    
    @staticmethod
    async def restore_agent_backup(backup_path: str) -> Dict[str, Any]:
        """Restore an agent from a backup file and create it in the database."""
        import os
        import json
        import uuid
        
        if not os.path.exists(backup_path):
            raise ValueError(f"Backup file not found: {backup_path}")
            
        try:
            # Load the backup file directly as JSON
            with open(backup_path, 'r') as f:
                agent_data = json.load(f)
                
            # Basic validation
            required_fields = ["name", "description"]
            for field in required_fields:
                if field not in agent_data:
                    raise ValueError(f"Missing required field '{field}' in backup file")
            
            # Create a new agent with the data from the backup
            original_name = agent_data["name"]
            
            # Remove any existing numeric suffix
            if '_' in original_name:
                base_name = original_name.split('_')[0]
            else:
                base_name = original_name
                
            # Generate a new UUID suffix - use just the last 8 characters
            uuid_suffix = uuid.uuid4().hex[-8:]
            new_name = f"{base_name}_{uuid_suffix}"
            
            # Get config data from the backup
            config_data = agent_data.get("llm_config", {})
            if not config_data and "config" in agent_data:
                # Try the top-level config if llm_config doesn't exist
                config_data = agent_data.get("config", {})
            
            # Create agent data in the format expected by AgentCreate
            agent_create = AgentCreate(
                name=new_name,
                description=agent_data.get("description", "Restored agent"),
                config={
                    "llm_type": config_data.get("llm_type", "OpenAILLM"),
                    "model": config_data.get("model", "gpt-3.5-turbo"),
                    "openai_key": config_data.get("openai_key", ""),
                    "temperature": config_data.get("temperature", 0.7),
                    "max_tokens": config_data.get("max_tokens", 150),
                    "top_p": config_data.get("top_p", 0.9),
                    "output_response": config_data.get("output_response", True),
                    "prompt": agent_data.get("system_prompt", "You are a helpful assistant.")
                },
                runtime_params=agent_data.get("runtime_params", {}),
                tags=agent_data.get("tags", ["restored"])
            )
            
            # Create the agent using AgentService
            created_agent = await AgentService.create_agent(
                agent_data=agent_create,
                user_id=None  # User ID is not available from the backup
            )
            
            return {
                "success": True,
                "message": f"Agent restored from {backup_path} with new name {new_name}",
                "agent_id": str(created_agent["_id"]),
                "backup_path": backup_path,
                "agent_name": created_agent["name"]
            }
        except Exception as e:
            logger.error(f"Failed to restore agent from backup: {str(e)}")
            raise ValueError(f"Failed to restore agent from backup: {str(e)}")
    
    @staticmethod
    async def list_agent_backups(agent_id: str, backup_dir: str) -> List[Dict[str, Any]]:
        """List all backup files for an agent."""
        if not ObjectId.is_valid(agent_id):
            raise ValueError(f"Invalid agent ID: {agent_id}")
            
        # Get the agent from database
        agent = await Database.agents.find_one({"_id": ObjectId(agent_id)})
        if not agent:
            raise ValueError(f"Agent with ID {agent_id} not found")
            
        try:
            import os
            import glob
            
            # Create backup directory if it doesn't exist
            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir)
            
            # Get all backup files for this agent
            backup_pattern = os.path.join(backup_dir, f"{agent['name']}_*.json")
            backup_files = glob.glob(backup_pattern)
            
            # Get file details
            backups = []
            for file_path in backup_files:
                file_stat = os.stat(file_path)
                backups.append({
                    "path": file_path,
                    "created_at": datetime.fromtimestamp(file_stat.st_ctime),
                    "size": file_stat.st_size
                })
            
            # Sort by creation date, newest first
            backups.sort(key=lambda x: x["created_at"], reverse=True)
            
            return backups
        except Exception as e:
            logger.error(f"Failed to list agent backups: {str(e)}")
            raise ValueError(f"Failed to list agent backups: {str(e)}")
    
    @staticmethod
    async def backup_all_agents(backup_dir: str) -> Dict[str, Any]:
        """
        Backup all agents in the AgentManager to the specified directory.
        
        Args:
            backup_dir: Directory where agent backups will be stored
            
        Returns:
            Dict with results of the batch backup operation
        """
        import os
        
        # Create backup directory if it doesn't exist
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
            
        # Get all agent names from the AgentManager
        agent_names = agent_manager.list_agents()
        
        results = []
        success_count = 0
        error_count = 0
        
        # Backup each agent in the AgentManager
        for agent_name in agent_names:
            try:
                # Find the agent in the database
                agent = await Database.agents.find_one({"name": agent_name})
                if not agent:
                    results.append({
                        "success": False,
                        "agent_name": agent_name,
                        "error": f"Agent '{agent_name}' found in AgentManager but not in database"
                    })
                    error_count += 1
                    continue
                
                agent_id = str(agent["_id"])
                backup_path = os.path.join(backup_dir, f"{agent_name}_backup.json")
                
                result = await AgentBackupService.save_agent_backup(agent_id, backup_path)
                results.append(result)
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to back up agent {agent_name}: {str(e)}")
                results.append({
                    "success": False,
                    "agent_name": agent_name,
                    "error": str(e)
                })
                error_count += 1
                
        return {
            "success": error_count == 0,
            "total": len(agent_names),
            "successful": success_count,
            "failed": error_count,
            "backup_dir": backup_dir,
            "results": results
        }
        
    @staticmethod
    async def backup_agents(agent_ids: List[str], backup_dir: str) -> Dict[str, Any]:
        """
        Backup specific agents to the specified directory, only if they exist in the AgentManager.
        
        Args:
            agent_ids: List of agent IDs to back up
            backup_dir: Directory where agent backups will be stored
            
        Returns:
            Dict with results of the batch backup operation
        """
        import os
        
        # Create backup directory if it doesn't exist
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
            
        results = []
        success_count = 0
        error_count = 0
        
        # Get all agent names in the AgentManager
        agent_names_in_manager = agent_manager.list_agents()
        
        # Backup each specified agent
        for agent_id in agent_ids:
            try:
                # Get agent from DB to get its name
                agent = await Database.agents.find_one({"_id": ObjectId(agent_id)})
                if not agent:
                    results.append({
                        "success": False,
                        "agent_id": agent_id,
                        "error": f"Agent with ID {agent_id} not found in database"
                    })
                    error_count += 1
                    continue
                    
                agent_name = agent["name"]
                
                # Check if agent is in the AgentManager
                if agent_name not in agent_names_in_manager:
                    results.append({
                        "success": False,
                        "agent_id": agent_id,
                        "agent_name": agent_name,
                        "error": f"Agent '{agent_name}' is not in the AgentManager (not running)"
                    })
                    error_count += 1
                    continue
                
                backup_path = os.path.join(backup_dir, f"{agent_name}_backup.json")
                
                result = await AgentBackupService.save_agent_backup(agent_id, backup_path)
                results.append(result)
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to back up agent {agent_id}: {str(e)}")
                results.append({
                    "success": False,
                    "agent_id": agent_id,
                    "error": str(e)
                })
                error_count += 1
                
        return {
            "success": error_count == 0,
            "total": len(agent_ids),
            "successful": success_count,
            "failed": error_count,
            "backup_dir": backup_dir,
            "results": results
        }
        
    @staticmethod
    async def restore_agents_from_files(backup_files: List[str]) -> Dict[str, Any]:
        """
        Restore multiple agents from a list of backup files.
        
        Args:
            backup_files: List of paths to agent backup files
            
        Returns:
            Dict with results of the batch restore operation
        """
        results = []
        success_count = 0
        error_count = 0
        
        # Restore each agent from the provided backup files
        for backup_path in backup_files:
            try:
                result = await AgentBackupService.restore_agent_backup(backup_path)
                results.append(result)
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to restore agent from {backup_path}: {str(e)}")
                results.append({
                    "success": False,
                    "backup_path": backup_path,
                    "error": str(e)
                })
                error_count += 1
                
        return {
            "success": error_count == 0,
            "total": len(backup_files),
            "successful": success_count,
            "failed": error_count,
            "results": results
        }
        
    @staticmethod
    async def restore_all_agents_from_directory(backup_dir: str) -> Dict[str, Any]:
        """
        Restore all agents from backup files in a directory.
        
        Args:
            backup_dir: Directory containing agent backup files
            
        Returns:
            Dict with results of the batch restore operation
        """
        import os
        import glob
        
        # Check if directory exists
        if not os.path.exists(backup_dir):
            raise ValueError(f"Backup directory not found: {backup_dir}")
            
        # Find all JSON files in the directory
        backup_files = glob.glob(os.path.join(backup_dir, "*.json"))
        
        if not backup_files:
            return {
                "success": True,
                "total": 0,
                "message": f"No backup files found in {backup_dir}"
            }
            
        # Restore all agents from the found backup files
        return await AgentBackupService.restore_agents_from_files(backup_files)