import ast
import contextlib
import io
import importlib
import sys
import os
import traceback
from typing import List, Set, Optional, Union, Dict, Any
from .interpreter_base import BaseInterpreter

class InterpreterPython(BaseInterpreter):
    def get_tool_info(self):
        return {
            "description": """The Python Interpreter Tool provides a secure execution environment for running Python code. 
            It performs static analysis using AST to detect unauthorized imports and security risks before execution. 
            The tool allows execution of both inline code snippets and external script files while enforcing strict security policies.
            
            The tool ensures:
            - Python code is analyzed using AST to detect and restrict unauthorized imports.
            - Scripts are executed in a controlled environment with predefined security constraints.
            - Only explicitly allowed imports are permitted, blocking unsafe system-wide libraries.
            - Project-specific modules can be dynamically loaded, ensuring local accessibility.
            - Execution output, including potential security violations or runtime errors, is captured and returned.
            """,
            
            "inputs": {
                "project_path": {
                    "type": "str",
                    "description": "The root directory of the project where Python scripts are located.",
                    "required": True
                },
                "allowed_imports": {
                    "type": "Optional[Set[str]]",
                    "description": "A set of module names that are explicitly allowed for import. If not provided, only local project modules are accessible.",
                    "required": False
                },
                "code": {
                    "type": "str",
                    "description": "The Python code snippet to be executed after safety checks.",
                    "required": True
                },
                "file_path": {
                    "type": "str",
                    "description": "The file path of the Python script to be executed.",
                    "required": True
                }
            },
            
            "outputs": {
                "execution_result": {
                    "type": "str",
                    "description": "The output of the executed code or an error message if execution fails."
                },
                "violations": {
                    "type": "list[str]",
                    "description": "A list of detected security violations, if any."
                }
            },
            
            "functionality": """Methods and their functionality:
            - `_analyze_code(code: str)`: Parses and analyzes the code using AST to detect restricted imports.
            - `_execute_import(import_module: ast.Import)`: Handles `import` statements, enforcing security policies.
            - `_execute_import_from(import_from: ast.ImportFrom)`: Manages `from module import name` statements securely.
            - `_check_project(module: Union[ast.Import, ast.ImportFrom])`: Verifies and imports project-specific modules.
            - `_extract_definitions(module_name: str, path: str, potential_names: Optional[Set[str]])`: Extracts functions and classes from a given module while ensuring safety.
            - `execute(code: str)`: Runs Python code after security checks, capturing output or errors.
            - `execute_script(file_path: str)`: Reads a script file and executes it securely.""",
            
            "interface": "execute(code: str) -> dict with key 'execution_result' (str) or 'violations' (list of str)"
        }


    project_path:str = ""
    directory_names:List[str] = []
    allowed_imports:Set[str] = set()
    namespace:Dict[str, Any] = {}

    def _get_file_and_folder_names(self, target_path: str) -> List[str]:
        """Retrieves the names of files and folders (without extensions) in a given directory.
        Args:
            target_path (str): Path to the target directory.
        Returns:
            List[str]: List of file and folder names (excluding extensions).
        """
        names = []
        for item in os.listdir(target_path):
            name, _ = os.path.splitext(item)  # Extract filename without extension
            names.append(name)
        return names

    def _extract_definitions(self, module_name: str, path: str, potential_names: Optional[Set[str]] = None) -> List[str]:
        """Extracts function and class definitions from a module file while ensuring safety.
        Args:
            module_name (str): The name of the module.
            path (str): The file path of the module.
            potential_names (Optional[Set[str]]): The specific functions/classes to import (for ImportFrom).
        Returns:
            List[str]: A list of violations found during analysis. An empty list indicates no issues.
        """
        if path in self.namespace:  # Avoid re-importing if already processed
            return []
        
        try:
            # Attempt to dynamically load the module
            module_spec = importlib.util.spec_from_file_location(module_name, path)
            loaded_module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(loaded_module)

            # Register the module in self.namespace
            self.namespace[module_name] = loaded_module

        except Exception:
            exc_type, exc_value, exc_tb = sys.exc_info()
            return ["".join(traceback.format_exception(exc_type, exc_value, exc_tb))]

        
        # Read the module file to perform code analysis
        with open(path, "r", encoding="utf-8") as f:
            code = f.read()

        # Perform safety check before adding functions/classes
        violations = self._analyze_code(code)
        if violations:
            return violations  # Stop execution if safety violations are detected

        # Parse the AST to extract function and class names
        tree = ast.parse(code)
        available_symbols = {}

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                available_symbols[node.name] = node  # Store detected functions/classes

        # Dynamically load specific functions/classes if requested
        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if potential_names is None:
                # Import all detected functions/classes
                for name in available_symbols:
                    if hasattr(module, name):
                        self.namespace[name] = getattr(module, name)
            else:
                # Import only specified functions/classes
                for name in potential_names:
                    if name in available_symbols and hasattr(module, name):
                        self.namespace[name] = getattr(module, name)
                    else:
                        violations.append(f"Function or class '{name}' not found in {module_name}")

        except Exception:
            exc_type, exc_value, exc_tb = sys.exc_info()
            return ["".join(traceback.format_exception(exc_type, exc_value, exc_tb))]


        return violations

    def _check_project(self, module: Union[ast.Import, ast.ImportFrom]) -> List[str]:
        """Checks and imports a local project module while ensuring safety.

        Args:
            module (Union[ast.Import, ast.ImportFrom]): The AST import node representing the module.

        Returns:
            List[str]: A list of violations found during analysis.
        """
        
        if isinstance(module, ast.Import):
            module_name = module.name
            potential_names = None  # Full module import
        else:
            module_name = module.module
            potential_names = {name.name for name in module.names}  # Selective import

        # Construct the module file path based on project structure
        if len(module_name.split(".")) > 1:
            module_path = os.path.join(self.project_path, *module_name.split(".")) + ".py"
        else:
            module_path = os.path.join(self.project_path, module_name + ".py")

        # Attempt to safely extract functions/classes
        if os.path.exists(module_path):
            violations = self._extract_definitions(module_name, module_path, potential_names)
        else:
            return [f"Module not found: {module_name}"]

        if violations:
            return violations  # Stop execution if safety violations are detected

        # Dynamically load the module and update self.namespace
        try:
            module_spec = importlib.util.spec_from_file_location(module_name, module_path)
            loaded_module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(loaded_module)

            # Register the module in self.namespace
            self.namespace[module_name] = loaded_module
            
        except Exception:
            exc_type, exc_value, exc_tb = sys.exc_info()
            return ["".join(traceback.format_exception(exc_type, exc_value, exc_tb))]

        return violations

    def _execute_import(self, import_module: ast.Import) -> List[str]:
        """Processes an import statement, verifying permissions and adding modules to the namespace.

        Args:
            import_module (ast.Import): The AST node representing an import statement.

        Returns:
            List[str]: A list of violations found during import handling.
        """
        violations = []
        
        for module in import_module.names:
            # Check if the module is part of the project directory (local module)
            if module.name.split(".")[0] in self.directory_names:
                violations += self._check_project(module)
                continue

            # Check if the import is explicitly allowed
            if module.name not in self.allowed_imports:
                violations.append(f"Unauthorized import: {module.name}")
                return violations

            # Attempt to import the module
            try:
                alias = module.asname or module.name
                imported_module = importlib.import_module(module.name)
                self.namespace[alias] = imported_module
            except ImportError:
                exc_type, exc_value, exc_tb = sys.exc_info()
                violations.append("".join(traceback.format_exception(exc_type, exc_value, exc_tb)))

        return violations

    def _execute_import_from(self, import_from: ast.ImportFrom) -> List[str]:
        """Processes a 'from module import name' statement, ensuring safety and adding modules to the namespace.

        Args:
            import_from (ast.ImportFrom): The AST node representing an 'import from' statement.

        Returns:
            List[str]: A list of violations found during import handling.
        """
        # Ensure that relative imports (e.g., 'from . import') are not allowed
        if import_from.module is None:
            return ["'from . import' is not supported."]

        # Check if the module is a part of the project directory (local module)
        if import_from.module.split(".")[0] in self.directory_names:
            return self._check_project(import_from)

        # Ensure that the module is explicitly allowed
        if import_from.module not in self.allowed_imports:
            return [f"Unauthorized import: {import_from.module}"]

        try:
            # Attempt to import the specified components from the module
            for import_name in import_from.names:
                imported_module = importlib.import_module(import_from.module)
                alias = import_name.asname or import_name.name
                self.namespace[alias] = getattr(imported_module, import_name.name)
            return []
        except ImportError:
            exc_type, exc_value, exc_tb = sys.exc_info()
            return ["".join(traceback.format_exception(exc_type, exc_value, exc_tb))]

    def _analyze_code(self, code: str) -> List[str]:
        """Parses and analyzes the code for import violations before execution.

        Args:
            code (str): The raw Python code to analyze.

        Returns:
            List[str]: A list of violations detected in the code.
        """
        violations = []

        try:
            # Parse the provided code into an Abstract Syntax Tree (AST)
            tree = ast.parse(code)

            # Traverse the AST and check for import violations
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    violations += self._execute_import(node)
                elif isinstance(node, ast.ImportFrom):
                    violations += self._execute_import_from(node)
        except SyntaxError:
            exc_type, exc_value, exc_tb = sys.exc_info()
            violations.append("".join(traceback.format_exception(exc_type, exc_value, exc_tb)))

        return violations

    def execute(self, code: str) -> None:
        """Analyzes and executes the provided Python code in a controlled environment.

        Args:
            code (str): The Python code to execute.

        Returns:
            str: The output of the executed code, or a list of violations if found.
        """
        self.visited_modules = {}
        self.namespace = {}

        # Change to the project directory and update sys.path for module resolution
        os.chdir(self.project_path)
        sys.path.insert(0, self.project_path)

        if self.allowed_imports:
            violations = self._analyze_code(code)
            if violations:
                return"\n".join(violations)
                

        # Capture standard output during execution
        stdout_capture = io.StringIO()
        with contextlib.redirect_stdout(stdout_capture):
            try:
                # Execute the code
                exec(code, {})
            except Exception:
                exc_type, exc_value, exc_tb = sys.exc_info()
                error_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
                return error_msg

        # Retrieve and return the captured output
        return stdout_capture.getvalue().strip()

    def execute_script(self, file_path: str, codetype: str) -> str:
        """
        Reads Python code from a file and executes it using the `execute` method.
        
        Args:
            file_path (str): The path to the Python file to be executed.
        Returns:
            str: The output of the executed code, or an error message if the execution fails.
        """
        
        if not os.path.isfile(file_path):
            return f"Error: File '{file_path}' does not exist."
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                code = file.read()
        except Exception as e:
            return f"Error reading file: {e}"
        
        return self.execute(code, codetype)