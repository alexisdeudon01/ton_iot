from typing import Dict, Type
from src.core.dag.task import Task

class TaskRegistry:
    _tasks: Dict[str, Type[Task]] = {}

    @classmethod
    def register(cls, name: str):
        def wrapper(task_cls: Type[Task]):
            cls._tasks[name] = task_cls
            return task_cls
        return wrapper

    @classmethod
    def get_task_cls(cls, name: str) -> Type[Task]:
        if name not in cls._tasks:
            raise ValueError(f"Task {name} not found in registry")
        return cls._tasks[name]
