import os
from typing import Optional

TaskDict = dict[str, "Task"]

class Task:
    """
    Base class for all tasks. Subclasses must implement the download and prepare
    methods.
    """
    _downloaded: Optional[str] = None
    _prepared: Optional[str] = None

    def __init_subclass__(cls) -> None:
        assert cls.download, "Task subclass must implement download method"
        assert cls.prepare, "Task subclass must implement prepare method"
        assert cls.name != "base", "Task subclass must have a name"

    # interface

    name: str = "base"

    def download(self, output_dir: str) -> str:
        raise NotImplementedError("Method not implemented")

    def prepare(self, prior_tasks: TaskDict) -> str: 
        raise NotImplementedError("Method not implemented")

    # getters and setters

    @property
    def downloaded(self) -> str:
        assert self._downloaded, f"{self.name} task has to be downloaded to prepare this task."
        return self._downloaded  # type: ignore

    @downloaded.setter
    def downloaded(self, value: str):
        self._downloaded = value

    @property
    def prepared(self) -> str:
        assert self._prepared, f"{self.name} task has to be prepared to prepare this task."
        return self._prepared

    @prepared.setter
    def prepared(self, value: str):
        self._prepared = value

class PrepareOnlyTask(Task):
    """
    Task that does not require downloading.
    """

    name = "prepare_only"

    def __init_subclass__(cls) -> None:
        assert cls.prepare, "PrepareOnlyTask subclass must implement prepare method"
        assert cls.name != "prepare_only", "PrepareOnlyTask subclass must have a name"

    def download(self, output_dir: str) -> str:
        self.downloaded = os.path.join(output_dir, f"{self.name}.dummy")
        return self.downloaded
