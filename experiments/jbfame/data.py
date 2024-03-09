#!/usr/bin/env python

import subprocess
from typing import Callable, Optional
import os
from pathlib import Path
from itertools import product
import warnings

import art
from tqdm import tqdm
import pandas as pd

def _wget(link: str, filename: str, output_dir: Optional[str] = None) -> str:
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    subprocess.run(["wget", "-qO", filename, link], cwd=output_dir)

def _download_dan(output_dir: str) -> str:
    dan_link = "https://raw.githubusercontent.com/verazuo/jailbreak_llms/main/data/jailbreak_prompts.csv"
    dan_path = "dan.csv"
    _wget(dan_link, dan_path, output_dir)
    return os.path.join(output_dir, dan_path)

def _download_null(output_dir: str) -> str:
    null_link = "https://raw.githubusercontent.com/verazuo/jailbreak_llms/main/data/questions.csv"
    null_path = "null.csv"
    _wget(null_link, null_path, output_dir)
    return os.path.join(output_dir, null_path)

def _prepare_null(downloaded_task: dict[str, str], prepared_task: dict[str, str]) -> str:
    """
    Extract question and question id from the dataset. This dataset is used to
    supply the other datasets with questions.
    """
    null_df = pd.read_csv(downloaded_task["null"])

    null_df = pd.DataFrame({ 
            "prompt": null_df["question"].to_list(),
            "q_id": null_df.index.to_list()
    })

    null_out = downloaded_task["null"].replace(".csv", ".parquet")
    null_df.to_parquet(null_out, index=False)

    return null_out

def _prepare_dan(downloaded_task: dict[str, str], prepared_task: dict[str, str]) -> str:
    dan_df = pd.read_csv(downloaded_task["dan"])
    null_df = pd.read_parquet(prepared_task["null"])
    
    dan_df = pd.DataFrame(
        product(dan_df["prompt"], zip(null_df["prompt"], null_df["q_id"])) , columns=["prompt", "question"]
    )
    dan_df[["question", "q_id"]] = pd.DataFrame(dan_df["question"].to_list(), index=dan_df.index)
    dan_df["prompt"] = dan_df[["prompt", "question"]].agg("\n".join, axis="columns")

    dan_df.drop(columns=["question"], inplace=True)

    dan_out = downloaded_task["dan"].replace(".csv", ".parquet")
    dan_df.to_parquet(dan_out, index=False)

    return dan_out

def _prepare_aart(downloaded_task: dict[str, str], prepared_task: dict[str, str]) -> str:
    def remove_words(text, words):
        return filter(lambda token: token.lower() not in words, text.split())

    def substitute(text, word_map):
        return [ text.replace(word, mapped_word) for word, mapped_word in word_map.items() ]

    def find_keywords_spacy(sentences: pd.Series):
        try:
            import spacy
            en = spacy.load('en_core_web_sm')
        except ImportError:
            raise ImportError("Cannot find scapy, please install with pip install scapy to use this task.")
        except OSError:
            raise OSError("Cannot find the model, please download with python -m spacy download en_core_web_sm")

        return sentences.apply(lambda sentence: remove_words(sentence, en.Defaults.stop_words))
        
    null_df = pd.read_parquet(prepared_task["null"])

    # find keywords and transform into ascii_art
    null_df["ascii_map"] = find_keywords_spacy(null_df["prompt"]).apply(lambda keywords: { 
        keyword : "\n" + art.text2art(keyword) for keyword in keywords 
    })

    # create new dataframe where questions have their keywords replaced by ascii art
    aart_df = pd.DataFrame(
        zip(
            null_df.apply(lambda row: substitute(row["prompt"], row["ascii_map"]), axis="columns").explode(ignore_index=True),
            null_df.apply(lambda row: len(row["ascii_map"]) * [row["q_id"]], axis="columns").explode(ignore_index=True)
        ), columns=["prompt", "q_id"],
    )
    aart_out = downloaded_task["aart"].replace(".csv", ".parquet")
    aart_df.to_parquet(aart_out, index=False)

    return aart_out

def _check_task(task: str):
    if task not in _download_task.keys():
        raise ValueError(f"Task {task} is not available.")

def _check_tasks(tasks: list[str]):
    [ _check_task(task) for task in tasks ]

_download_task: dict[str, Callable[[str], str]] = {
    "dan": _download_dan,
    "null": _download_null,
    "aart": lambda output_dir: os.path.join(output_dir, "aart.csv")  # aart is generated from null
}

_prepare_task: dict[str, Callable[[dict, dict], str]] = {
    "dan": _prepare_dan,
    "null": _prepare_null,
    "aart": _prepare_aart,
    # "uaa": lambda thing: "lol",
}

try:
    assert _prepare_task.keys() == _download_task.keys()
except AssertionError:
    print("When inserting new tasks, each task should have a 'download' and preparation method. Even if the task is synthetic")
    print(f"download_task: {_download_task.keys()}\nprepare_task: {_prepare_task.keys()}")
    exit(1)

def available_tasks():
    return list(_download_task.keys())

def download(tasks: list[str], output_dir: str) -> dict[str, str]:
    task_paths = {}

    # dowload selected tasks in output_dir
    for task in (pbar := tqdm(tasks, desc="Downloading tasks")):
        pbar.set_postfix_str(task)
        task_paths[task] = _download_task[task](output_dir)

    return task_paths

def prepare(downloaded_task: dict[str, str], cleanup: bool = True) -> dict[str, str]:
    if "null" not in downloaded_task:
        print("Null task has to be present to build other tasks.")

    prepared_task = {}
      
    # prepare all tasks that have been downloaded
    for task in (pbar := tqdm(downloaded_task.keys(), desc="Preparing tasks")): 
        pbar.set_postfix_str(task)
        prepared_task[task] = _prepare_task[task](downloaded_task, prepared_task)

    # cleanup old downloaded files
    if cleanup:
        for old_file in downloaded_task.values():
            if os.path.exists(old_file):
                os.remove(old_file)

    return prepared_task

def download_and_prepare(tasks: Optional[list[str]] = None, output_dir="data", cleanup: bool = True) -> dict[str, str]:
    if not tasks:
        tasks = list(_download_task.keys())
    if "null" not in tasks:
        tasks = ["null"] + tasks

    # throws value error if tasks are present that cannot be downloaded
    _check_tasks(tasks)

    # download and prepare tasks, cleanup if chosen
    task_paths = download(tasks, output_dir)
    task_paths = prepare(task_paths, cleanup)

    # return final tasks; allows easy reading through, for example, pandas
    return task_paths

if __name__=="__main__":
    warnings.filterwarnings("ignore")

    import typer
    typer.run(download_and_prepare)
