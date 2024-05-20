from jbfame.tasks.null import Null
from jbfame.tasks.dan import Dan
from jbfame.tasks.aart import AArt
from jbfame.tasks.adan import ADan
from jbfame.tasks.wiki import Wiki
from jbfame.tasks.enc_b64 import EncB64

from jbfame.tasks.base import Task, TaskDict

all_tasks: dict[str, type[Task]] = {
    Null.name: Null,
    Dan.name: Dan,
    AArt.name: AArt,
    ADan.name: ADan,
    Wiki.name: Wiki,
    EncB64.name: EncB64,
}

__all__ = [
    "Null",
    "Dan",
    "AArt",
    "Wiki",
    "ADan",
    "EncB64",
    "TaskDict",
]
