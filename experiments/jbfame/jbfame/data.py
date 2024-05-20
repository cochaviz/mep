#!/usr/bin/env python

import os
from typing import Optional
import warnings

from tqdm import tqdm

from jbfame.tasks import TaskDict, all_tasks

def available_tasks():
    return list(all_tasks.keys())

def download(requested_tasks: list[str], output_dir: str) -> TaskDict:
    assert "null" in requested_tasks, "null task has to be present to build other tasks."

    tasks = { 
        task_name : all_tasks[task_name]() 
             for task_name in requested_tasks 
        }

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

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

    # cleanup old downloaded files
    if cleanup:
        for old_file in map(lambda task: task.downloaded, tasks.values()):
            if os.path.exists(old_file) and not os.path.isdir(old_file):
                os.remove(old_file)

    return tasks

def download_and_prepare(
    tasks: Optional[list[str]] = None, 
    output_dir="data", 
    cleanup: bool = True, 
) -> dict[str, str]:
    tasks = tasks or available_tasks()

    if "null" not in tasks:
        tasks = ["null"] + tasks

    # download and prepare tasks, cleanup if chosen
    tasks_dict = download(tasks, output_dir)
    tasks_dict = prepare(tasks_dict, cleanup)

    # return final tasks; allows easy reading through, for example, pandas
    task_paths = {task_name: task.downloaded for task_name, task in tasks_dict.items()}
    return task_paths

if __name__=="__main__":
    warnings.filterwarnings("ignore")

    import typer
    typer.run(download_and_prepare)
