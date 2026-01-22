from typing import List, Dict, Set
from src.core.dag.task import Task

class DAGGraph:
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.dependencies: Dict[str, Set[str]] = {}

    def add_task(self, task: Task, depends_on: List[str] = None):
        self.tasks[task.name] = task
        self.dependencies[task.name] = set(depends_on or [])

    def get_execution_order(self) -> List[str]:
        order = []
        visited = set()
        temp_visited = set()

        def visit(name):
            if name in temp_visited:
                raise ValueError(f"Circular dependency detected at {name}")
            if name not in visited:
                temp_visited.add(name)
                for dep in self.dependencies.get(name, []):
                    visit(dep)
                temp_visited.remove(name)
                visited.add(name)
                order.append(name)

        for task_name in self.tasks:
            if task_name not in visited:
                visit(task_name)
        return order

    def to_mermaid(self) -> str:
        lines = ["graph TD"]
        for task_name, deps in self.dependencies.items():
            for dep in deps:
                lines.append(f"  {dep} --> {task_name}")
            if not deps:
                lines.append(f"  {task_name}")
        return "\n".join(lines)
