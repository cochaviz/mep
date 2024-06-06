import os
from typing import Optional

TaskDict = dict[str, "Task"]

import logging
logger = logging.getLogger(__name__)

class Task:
    """
    Base class for all tasks. Subclasses must implement the download and prepare
    methods.
    """
    _downloaded: Optional[str] = None
    _prepared: Optional[str] = None

    def __init_subclass__(cls) -> None:
        assert cls._download, "Task subclass must implement download method"
        assert cls._prepare, "Task subclass must implement prepare method"
        assert cls.name != "base", "Task subclass must have a name"

    # interface

    name: str = "base"

    def download(self, output_dir: str) -> str:
        if not self._downloaded and not self._prepared:
            logger.debug(f"No downloaded or prepared file found for {self.name}. Downloading.")
            self.downloaded = self._download(output_dir)

        # it doesn't matter which one is returned, as long as 
        # it's one of them.
        try: 
            return self.downloaded 
        except AssertionError:
            logger.info("No downloaded file found. Returning prepared file.")
            return self.prepared
    
    def _download(self, output_dir: str) -> str:
        raise NotImplementedError("Method not implemented")

    def prepare(self, prior_tasks: TaskDict) -> str: 
        if not self._prepared:
            logger.debug(f"No prepared file found for {self.name}. Preparing.")
            self.prepared = self._prepare(prior_tasks)
        return self.prepared

    def _prepare(self, prior_tasks: TaskDict) -> str:
        raise NotImplementedError("Method not implemented") 
        
    def populate(self, output_dir: str) -> "Task":
        if not os.path.exists(output_dir):
            return self

        matches = list(filter(
                lambda name: 
                    name.startswith(f"{self.name}") and name.endswith(".parquet"), 
                    os.listdir(output_dir)
            ))
        if len(matches) > 0:
            logger.info(f"Found prepared '.parquet' file for {self.name}.")
            self.prepared = os.path.join(output_dir, matches[0])

        return self

    # getters and setters

    @property
    def downloaded(self) -> str:
        assert self._downloaded or self._prepared, f"{self.name} task is not downloaded or prepared."
        
        if not self._downloaded:
            return f"{self.name}.dummy"

        return self._downloaded

    @downloaded.setter
    def downloaded(self, value: str):
        self._downloaded = value

    @property
    def prepared(self) -> str:
        assert self._prepared, f"{self.name} task is not prepared."
        return self._prepared

    @prepared.setter
    def prepared(self, value: str):
        self._prepared = value

    def __str__(self) -> str:
        return f"Task {self.name} downloaded at {self._downloaded or "N/A"} and prepared at {self._prepared or "N/A"}"

class PrepareOnlyTask(Task):
    """
    Task that does not require downloading.
    """

    name = "base_prepare"

    def __init_subclass__(cls) -> None:
        assert cls.prepare, "PrepareOnlyTask subclass must implement prepare method"
        assert cls.name != "base_prepare", "PrepareOnlyTask subclass must have a name"

    def download(self, output_dir: str) -> str:
        logger.info(f"Skipping download for {self.name} (prepare-only task).")
        self.downloaded = os.path.join(output_dir, f"{self.name}.dummy")
        return self.downloaded
