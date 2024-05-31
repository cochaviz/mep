#!/usr/bin/env python

import os
from typing import Optional

from tqdm import tqdm
from datasets import DatasetDict, Dataset
from pyarrow import parquet

from jbfame.tasks import TaskDict, all_tasks

import logging
logger = logging.getLogger(__name__)

def _task_paths_to_datasetdict(task_paths: dict[str, str]) -> DatasetDict:
    """
    Reads a parquet file and returns it as a pandas DataFrame.
    """
    datasets = DatasetDict()

    # FIXME This should be able to be done in a single line, but for whatever
    # reason, Datasets.read_parquet does not work as expected.
    for task, path in task_paths.items():
        datasets[task] = Dataset(parquet.read_table(path))

    return datasets
        
def available_tasks():
    return list(all_tasks.keys())

def check_tasks(tasks: list[str]) -> bool:
    return set(tasks).issubset(available_tasks())


def download(tasks: TaskDict, output_dir: str) -> TaskDict:
    assert "null" in tasks, "null task has to be present to build other tasks."

    logger.info(f"Downloading tasks to {output_dir}.")
    logger.debug(f"Tasks to download: {tasks.keys()}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # dowload selected tasks in output_dir
    for task_name, task in (pbar := tqdm(tasks.items(), desc="Downloading tasks")):
        pbar.set_postfix_str(task_name)
        task.download(output_dir)

    return tasks

def prepare(tasks: TaskDict, cleanup: bool = True) -> TaskDict:
    assert "null" in tasks, "null task has to be present to build other tasks."

    logger.info(f"Preparing tasks.")

    # prepare all tasks that have been downloaded
    for task_name, task in (pbar := tqdm(tasks.items(), desc="Preparing tasks")): 
        pbar.set_postfix_str(task_name)
        task.prepare(tasks)

    if not cleanup:
        logger.info("Skipping cleanup.")
        return tasks

    logger.info("Cleaning up old downloaded files.")
    # cleanup old downloaded files
    for old_file in map(lambda task: task.downloaded, tasks.values()):
        if os.path.exists(old_file) and not os.path.isdir(old_file):
            logger.debug(f"Removing {old_file}.")
            os.remove(old_file)

    return tasks

def download_and_prepare(
    tasks: Optional[list[str]] = None, 
    output_dir="data", 
    cleanup: bool = True, 
) -> DatasetDict:
    tasks = tasks or available_tasks()

    if not check_tasks(tasks):
        raise ValueError(f"Not all tasks in {tasks} are available. Available tasks are {available_tasks()}.")

    if "null" not in tasks:

        tasks = ["null"] + tasks

    # retrieve tasks and populate if already present
    task_dict: TaskDict = { 
        task_name : all_tasks[task_name]().populate(output_dir)
             for task_name in tasks
        }

    # download and prepare unprepared tasks, cleanup if chosen
    download(task_dict, output_dir)
    prepare(task_dict, cleanup)

    task_paths = { task_name: task.prepared for task_name, task in task_dict.items() }

    return _task_paths_to_datasetdict(task_paths)

if __name__=="__main__":
    import typer
    typer.run(download_and_prepare)
