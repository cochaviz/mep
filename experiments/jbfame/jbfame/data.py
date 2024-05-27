#!/usr/bin/env python

import os
from typing import Optional

from tqdm import tqdm
from datasets import DatasetDict, Dataset
from pyarrow import parquet

from jbfame.tasks import TaskDict, all_tasks

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

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # dowload selected tasks in output_dir
    for task_name, task in (pbar := tqdm(tasks.items(), desc="Downloading tasks")):
        pbar.set_postfix_str(task_name)
        task.download(output_dir)

    return tasks

def prepare(tasks: TaskDict, cleanup: bool = True) -> TaskDict:
    assert "null" in tasks, "null task has to be present to build other tasks."

    # prepare all tasks that have been downloaded
    for task_name, task in (pbar := tqdm(tasks.items(), desc="Preparing tasks")): 
        pbar.set_postfix_str(task_name)
        task.prepare(tasks)

    if not cleanup:
        return tasks

    # cleanup old downloaded files
    for old_file in map(lambda task: task.downloaded, tasks.values()):
        if os.path.exists(old_file) and not os.path.isdir(old_file):
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

    # # back-and-forth with lists because typer doesn't like sets as input
    # tasks_necessary = _necessary_tasks(tasks_requested, output_dir)
    # tasks_ready = list(set(tasks_requested) - set(tasks_necessary))

    # # check if output_dir exists and contains parquet files
    # if os.path.exists(output_dir) and tasks_necessary < tasks_requested:
    #     warnings.warn(f"Data directory {output_dir} already exists and contains parquet files. The tasks that have already been prepared will not be prepared again.")

    #     # if all tasks are already downloaded, we're good
    #     if len(tasks_necessary) == 0:
    #         warnings.warn(f"No tasks to download and prepare.")
    #         all_task_paths = _tasks_in_folder_to_path(tasks_requested, output_dir, "parquet")
    #         return _task_paths_to_datasetdict(all_task_paths)

    # retrieve tasks and populate if already present
    task_dict: TaskDict = { 
        task_name : all_tasks[task_name]().populate(output_dir)
             for task_name in tasks
        }

    # download and prepare unprepared tasks, cleanup if chosen
    download(task_dict, output_dir)
    prepare(task_dict, cleanup)

    # # return final tasks; allows easy reading through, for example, pandas
    # all_task_paths = { task_name: task.prepared for task_name, task in tasks_necessary_dict.items() }

    # # retrieve available tasks from output_dir and add them to the task_paths
    # all_task_paths.update(
    #     _tasks_in_folder_to_path(
    #         tasks_ready, 
    #         output_dir, 
    #         "parquet"
    #     ).items()
    # )

    task_paths = { task_name: task.prepared for task_name, task in task_dict.items() }

    return _task_paths_to_datasetdict(task_paths)

if __name__=="__main__":
    import typer
    typer.run(download_and_prepare)
