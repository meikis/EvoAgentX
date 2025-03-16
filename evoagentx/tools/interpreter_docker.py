import io
import shlex
import tarfile
import uuid
import docker
from pathlib import Path
from typing import ClassVar, Dict, Set, Optional
from .interpreter_base import BaseInterpreter

class DockerInterpreter(BaseInterpreter):
    """
    A Docker-based interpreter for executing Python, Bash, and R scripts in an isolated environment.
    """

    def get_tool_info(self):
        return {
            "description": """The Docker Interpreter Tool provides a secure and isolated environment for executing Python code inside a Docker container. 
            It ensures controlled execution by leveraging containerization, allowing scripts to run without affecting the host system. 
            The tool supports mounting local directories into the container, enabling seamless access to external files.
            
            The tool ensures:
            - Code execution inside a predefined Docker container with restricted access.
            - Support for executing scripts from inline code snippets or files in a mounted directory.
            - Standard output and error message capture for debugging and verification.
            - Optional user confirmation before executing scripts.
            - Secure transfer of files from the host system into the container environment.""",
            
            "inputs": {
                "require_confirm": {
                    "type": "bool",
                    "description": "If True, execution requires user confirmation.",
                    "required": False
                },
                "print_stdout": {
                    "type": "bool",
                    "description": "If True, prints standard output from the container execution.",
                    "required": False
                },
                "print_stderr": {
                    "type": "bool",
                    "description": "If True, prints standard error from the container execution.",
                    "required": False
                },
                "image_tag": {
                    "type": "str",
                    "description": "The Docker image tag used for container execution. Defaults to 'fundingsocietiesdocker/python3.9-slim'.",
                    "required": False
                },
                "dockerfile_path": {
                    "type": "str",
                    "description": "The path to the Dockerfile used for building the image if not found.",
                    "required": False
                },
                "host_directory": {
                    "type": "str",
                    "description": "A local directory that will be mounted into the container for execution.",
                    "required": False
                },
                "container_directory": {
                    "type": "str",
                    "description": "The corresponding directory inside the container where the host directory is mounted.",
                    "required": False
                },
                "code": {
                    "type": "str",
                    "description": "The Python code snippet to be executed inside the Docker container.",
                    "required": True
                },
                "code_type": {
                    "type": "str",
                    "description": "The programming language of the code being executed. Currently supports Python ('python', 'py', 'py3', 'python3').",
                    "required": True
                }
            },
            
            "outputs": {
                "execution_result": {
                    "type": "str",
                    "description": "The output of the executed code inside the container, including standard output and error messages."
                },
                "error": {
                    "type": "str",
                    "description": "An error message if execution fails inside the container."
                }
            },
            
            "functionality": """Methods and their functionality:
            - `_initialize_if_needed()`: Ensures the Docker container is initialized and the required image is available.
            - `_upload_directory_to_container(host_directory: str)`: Transfers files from a given host directory into the container.
            - `_create_file_in_container(content: str)`: Generates a temporary file inside the container for execution.
            - `_run_file_in_container(file: Path, code_type: str)`: Executes the specified file inside the container and retrieves output.
            - `execute(code: str, code_type: str)`: Runs Python code inside the Docker container, optionally confirming execution.
            - `_check_code_type(code_type: str)`: Validates and maps supported programming languages to the correct Docker execution command.""",
            
            "interface": "execute(code: str, code_type: str) -> dict with key 'execution_result' (str) or 'error' (str)"
        }

    _CODE_EXECUTE_CMD_MAPPING: ClassVar[Dict[str, str]] = {
        "python": "python {file_name}",
    }

    _CODE_TYPE_MAPPING: ClassVar[Dict[str, str]] = {
        "python": "python",
        "py3": "python",
        "python3": "python",
        "py": "python",
    }

    require_confirm:bool = False
    print_stdout:bool = True
    print_stderr:bool = True
    _container:docker.models.containers.Container = None
    image_tag:str = "fundingsocietiesdocker/python3.9-slim"
    dockerfile_path:str = "./docker/Dockerfile"
    host_directory:str = ""
    container_directory:str = "/home/app/"
    
    class Config:
        arbitrary_types_allowed = True  # Allow non-pydantic types like sets

    def __init__(self, **data):
        super().__init__(**data)
        self._client = docker.from_env()
        self._initialize_if_needed()
        if self.host_directory:
            self._upload_directory_to_container(self.host_directory)
        

    def __del__(self):
        if self._container is not None:
            self._container.remove(force=True)

    def _initialize_if_needed(self):
        if self._container is not None:
            return
        
        dockerfile_path = Path("./docker/Dockerfile")

        try:
            self._client.images.get(self.image_tag)
        except docker.errors.ImageNotFound:
            if not dockerfile_path.exists():
                raise FileNotFoundError(f"Dockerfile not found at {dockerfile_path}")
            
            self._client.images.build(path="./docker", tag=self.image_tag, rm=True, buildargs={})

        self._container = self._client.containers.run(
            self.image_tag, detach=True, command="tail -f /dev/null",working_dir="/home/app/"
        )

        # self.execute(, "python")

    def _upload_directory_to_container(self, host_directory: str):
        """
        Uploads all files and directories from the given host directory to /home/app/ in the container.

        :param host_directory: Path to the local directory containing files to upload.
        :param container_directory: Target directory inside the container (defaults to "/home/app/").
        """
        host_directory = Path(host_directory).resolve()
        if not host_directory.exists() or not host_directory.is_dir():
            raise FileNotFoundError(f"Directory not found: {host_directory}")

        tar_stream = io.BytesIO()
        
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            for file_path in host_directory.rglob("*"):
                if file_path.is_file():
                    # Ensure path is relative to the given directory
                    relative_path = file_path.relative_to(host_directory)
                    target_path = Path(self.container_directory) / relative_path
                    
                    tarinfo = tarfile.TarInfo(name=str(target_path.relative_to(self.container_directory)))
                    tarinfo.size = file_path.stat().st_size
                    with open(file_path, "rb") as f:
                        tar.addfile(tarinfo, f)

        tar_stream.seek(0)

        if self._container is None:
            raise RuntimeError("Container is not initialized.")

        self._container.put_archive(self.container_directory, tar_stream)

        # Ensure the uploaded directory is in sys.path for imports
        # self._container.exec_run(f"echo 'export PYTHONPATH=/home/app/:$PYTHONPATH' | sudo tee -a /etc/environment")

    def _create_file_in_container(self, content: str) -> Path:
        filename = str(uuid.uuid4())
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode='w') as tar:
            tarinfo = tarfile.TarInfo(name=filename)
            tarinfo.size = len(content)
            tar.addfile(tarinfo, io.BytesIO(content.encode('utf-8')))
        tar_stream.seek(0)

        if self._container is None:
            raise RuntimeError("Container is not initialized.")
        self._container.put_archive("/tmp", tar_stream)
        return Path(f"/tmp/{filename}")

    def _run_file_in_container(self, file: Path, code_type: str) -> str:
        code_type = self._check_code_type(code_type)
        command = shlex.split(self._CODE_EXECUTE_CMD_MAPPING[code_type].format(file_name=file.as_posix()))
        if self._container is None:
            raise RuntimeError("Container is not initialized.")
        result = self._container.exec_run(command, demux=True)

        stdout, stderr = result.output
        if self.print_stdout and stdout:
            print(stdout.decode())
        if self.print_stderr and stderr:
            print(stderr.decode())

        return stdout.decode() if stdout else "" + (stderr.decode() if stderr else "")

    def execute(self, code: str, code_type: str) -> str:
        if self.host_directory:
            code = f"import sys; sys.path.insert(0, '{self.container_directory}');" + code
        code_type = self._check_code_type(code_type)
        if self.require_confirm:
            confirmation = input(f"Confirm execution of {code_type} code? [Y/n]: ")
            if confirmation.lower() not in ["y", "yes", ""]:
                raise RuntimeError("Execution aborted by user.")
        
        
        file_path = self._create_file_in_container(code)
        return self._run_file_in_container(file_path, code_type)

    def _check_code_type(self, code_type: str) -> str:
        if code_type not in self._CODE_TYPE_MAPPING:
            raise ValueError(f"Unsupported code type: {code_type}")
        return self._CODE_TYPE_MAPPING[code_type]

    _CODE_EXECUTE_CMD_MAPPING: ClassVar[Dict[str, str]] = {
        "python": "python {file_name}",
    }

    _CODE_TYPE_MAPPING: ClassVar[Dict[str, str]] = {
        "python": "python",
        "py3": "python",
        "python3": "python",
        "py": "python",
    }
