#!/usr/bin/env python

import subprocess
from typing import Optional
import os
from pathlib import Path
from itertools import product
import warnings

import art
from tqdm import tqdm
import pandas as pd

def _wget(link: str, filename: str, output_dir: Optional[str] = None):
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    subprocess.run(["wget", "-qO", filename, link], cwd=output_dir)

def _download_dan(output_dir: str):
    dan_link = "https://raw.githubusercontent.com/verazuo/jailbreak_llms/main/data/jailbreak_prompts.csv"
    dan_path = "dan.csv"
    _wget(dan_link, dan_path, output_dir)
    return os.path.join(output_dir, dan_path)

def _download_null(output_dir: str):
    null_link = "https://raw.githubusercontent.com/verazuo/jailbreak_llms/main/data/questions.csv"
    null_path = "null.csv"
    _wget(null_link, null_path, output_dir)
    return os.path.join(output_dir, null_path)

def _prepare_null(null: str):
    """
    Extract question and question id from the dataset. This dataset is used to
    supply the other datasets with questions.
    """
    null_df = pd.read_csv(null)
    null_df = null_df[["question", "q_id"]]
    null_df.to_parquet(null.replace(".csv", ".parquet"), index=False)
    os.remove(null)

def _prepare_dan(dan: str, null: str):
    dan_df = pd.read_csv(dan)
    null_df = pd.read_csv(null)
    
    dan_df = pd.DataFrame(
        product(dan_df["prompt"], zip(null_df["question"], null_df["q_id"])) , columns=["prompt", "question"]
    )
    dan_df["q_id"] = dan_df["question"].apply(lambda values: values[1])
    dan_df["question"] = dan_df["question"].apply(lambda values: values[0])
    dan_df["prompt"] = dan_df[["prompt", "question"]].agg("\n".join, axis="columns")
    dan_df.drop(columns=["question"], inplace=True)

    dan_df.to_parquet(dan.replace(".csv", ".parquet"), index=False)
    os.remove(dan)

def _prepare_aart(null: str, aart: str):
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

        return sentences.apply(lambda question: remove_words(question, en.Defaults.stop_words))
        
    null_df = pd.read_csv(null)

    # find keywords and transform into ascii_art
    null_df["ascii_map"] = find_keywords_spacy(null_df["question"]).apply(lambda keywords: { 
        keyword : "\n" + art.text2art(keyword) for keyword in keywords 
        })

    # create new dataframe where questions have their keywords replaced by ascii art
    aart_df = pd.DataFrame(
        zip(
            null_df.apply(lambda row: substitute(row["question"], row["ascii_map"]), axis="columns").explode(ignore_index=True),
            null_df.apply(lambda row: len(row["ascii_map"]) * [row["q_id"]], axis="columns").explode(ignore_index=True)
        ), columns=["prompt", "q_id"],
    )
    aart_df.to_parquet(aart.replace(".csv", ".parquet"), index=False)
    assert not os.path.exists(aart)

def _check_task(task: str):
    if task not in download_task.keys():
        raise ValueError(f"Task {task} is not available.")

def _check_tasks(tasks: list[str]):
    [ _check_task(task) for task in tasks ]

download_task = {
    "dan": _download_dan,
    "null": _download_null,
    "aart": lambda output_dir: os.path.join(output_dir, "aart.csv")  # aart is generated from null
}

def download(tasks: list[str], output_dir: str):
    task_paths = {}

    for task in (pbar := tqdm(tasks, desc="Downloading tasks")):
        pbar.set_postfix_str(task)
        task_paths[task] = download_task[task](output_dir)

    return task_paths

def prepare(task_paths: dict[str, str], output_dir: str):
    if "null" not in task_paths:
        print("Null task has to be present to build other tasks.")
       
    for task in (pbar := tqdm(task_paths.keys(), desc="Preparing tasks")): 
        pbar.set_postfix_str(task)

        if task == "null":
            _prepare_null(task_paths["null"])
        if task == "dan":
            _prepare_dan(task_paths["dan"], task_paths["null"])
        if task == "aart":
            _prepare_aart(task_paths["null"], task_paths["aart"])

def download_and_prepare(tasks: Optional[list[str]] = None, output_dir="data"):
    if not tasks:
        tasks = list(download_task.keys())
    if "null" not in tasks:
        tasks.append("null")

    _check_tasks(tasks)

    task_paths = download(tasks, output_dir)
    prepare(task_paths, output_dir)

if __name__=="__main__":
    warnings.filterwarnings("ignore")

    import typer
    typer.run(download_and_prepare)
