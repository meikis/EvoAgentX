import unittest
from evoagentx.tools.interpreter_python import InterpreterPython
import os
import shutil
import tempfile

class TestInterpreterPython(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for the project
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project_path = os.path.join(self.temp_dir.name, "AGTChart")
        
        # Create the project directory structure
        os.makedirs(os.path.join(self.project_path, "test"), exist_ok=True)

        # Create evi.py
        with open(os.path.join(self.project_path, "evi.py"), "w") as f:
            f.write("""def func1():\n    return \"Hello from func1\"""")

        # Create test/other.py
        with open(os.path.join(self.project_path, "test", "other.py"), "w") as f:
            f.write("""\n\ndef func2():\n  return \"Hello from func2\"""")

        # Define allowed imports/functions
        self.allowed_imports = {"math", "numpy", "os", "sys"}  # Allowed modules (optional)
        self.interpreter = InterpreterPython(project_path=self.project_path, allowed_imports=self.allowed_imports)

    def tearDown(self):
        # Clean up the temporary directory
        self.temp_dir.cleanup()

    def test_execute_valid_code(self):
        # Sample code that imports from AGTChart
        test_code = """
import os
import math
from math import sqrt
import numpy as np
import sys

print('Hello World')
print(len([1, 2, 3]))
print("Testing Project Imports")
np.array([1, 2, 3])
print(sqrt(4))
"""
        result = self.interpreter.execute(test_code)
        self.assertIn("Hello World", result)
        self.assertIn("Testing Project Imports", result)

    def test_unauthorized_import(self):
        test_code = "import subprocess"
        result = self.interpreter.execute(test_code)
        self.assertIn("Unauthorized import", result)

    def test_syntax_error(self):
        test_code = "print('Hello World'"
        result = self.interpreter.execute(test_code)
        self.assertIn("SyntaxError", result)

    def test_runtime_error(self):
        test_code = "print(1/0)"
        result = self.interpreter.execute(test_code)
        self.assertIn("ZeroDivisionError", result)

if __name__ == '__main__':
    unittest.main()
