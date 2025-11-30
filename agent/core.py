from typing import Callable, Dict, Any


class Agent:
    """
    Minimal 'agent' core:
    - registers tools (functions)
    - receives input
    - decides which tool to call
    - returns result
    """

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.tools: Dict[str, Callable[..., Any]] = {}

    def add_tool(self, name: str, func: Callable[..., Any]):
        self.tools[name] = func

    def list_tools(self):
        return list(self.tools.keys())

    def call_tool(self, name: str, **kwargs):
        if name not in self.tools:
            raise ValueError(f"Tool '{name}' not registered.")
        return self.tools[name](**kwargs)

