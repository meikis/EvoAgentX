from .tool import Tool

class BaseInterpreter(Tool):

    def get_tool_info(self):
        """
        Returns the tool information by extending the base Tool class.
        """
        return super().get_tool_info()

    def get_tool_description(self):
        """
        Returns a brief description of the base interpreter tool.
        """
        return "BaseInterpreter is an abstract class for interpreters that execute code securely. It enforces safety checks before execution and serves as the foundation for language-specific interpreters."

    # def get_tool_info(self):
    #     return super().get_tool_info()

    def execute(self, code: str, codetype: str) -> str:
        """Checks the code for safety before execution."""
        pass
