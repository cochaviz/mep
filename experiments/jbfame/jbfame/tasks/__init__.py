from jbfame.tasks.wei import all_tasks as wei_tasks
from jbfame.tasks.custom import all_tasks as custom_tasks

from jbfame.tasks.base import Task, TaskDict
from jbfame.tasks.null import Null

all_tasks: dict[str, type[Task]] = {
    Null.name: Null,
    # **custom_tasks,
    **wei_tasks,
}

__all__ = [
    "Task",
    "TaskDict",
]
