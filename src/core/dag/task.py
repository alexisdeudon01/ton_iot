from abc import ABC, abstractmethod
from typing import List
from src.core.dag.context import DAGContext
from src.core.dag.result import TaskResult

class Task(ABC):
    def __init__(self, name: str, inputs: List[str] = None, outputs: List[str] = None):
        self.name = name
        self.inputs = inputs or []
        self.outputs = outputs or []

    @abstractmethod
    def run(self, context: DAGContext) -> TaskResult:
        pass
