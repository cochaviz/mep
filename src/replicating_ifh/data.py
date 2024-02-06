#!/usr/bin/env python3

import glob
import os
from enum import Enum

import pandas as pd
from datasets import load_dataset_builder

from fine_tuners import AvailableFineTuner

fs_glue_location = os.path.join(os.getcwd(), "data/fs_glue")
fs_glue_default_location = os.path.join(fs_glue_location, "default")

glue_task_names = [ "cola", "mrpc", "qqp", "mnli", "qnli", "rte" , "sst2" ]
superglue_task_names = [ "boolq", "cb", "copa" , "wic" ]
fs_glue_task_names = glue_task_names + superglue_task_names

class AvailableDataset(str, Enum):
    none = "none"
    fs_glue = "fs_glue"
    fs_nli = "fs_nli"
    all = "all"

def _download_fs_glue():
    if os.path.isdir(fs_glue_default_location):
        print(f"Datasets already exist in {fs_glue_default_location}.")
        return

    print(f"Downloading datasets into {fs_glue_default_location}...")

    for task in glue_task_names:
        builder = load_dataset_builder("glue", task)
        builder.download_and_prepare(
                f"{fs_glue_default_location}/{task}", 
                file_format="parquet"
            )

    for task in superglue_task_names:
        builder = load_dataset_builder(f"super_glue", task)
        builder.download_and_prepare(
                f"{fs_glue_default_location}/{task}", 
                file_format="parquet"
            )

    # remove lock files
    for file in glob.glob(f"{fs_glue_default_location}/*.lock"):
        os.remove(file)

    # rename files
    for file in glob.glob(f"{fs_glue_default_location}/*/*.parquet"):
        dirname = os.path.dirname(file)
        if "train" in file:
            new_file = os.path.join(dirname, "train.parquet")
        if "test" in file:
            new_file = os.path.join(dirname, "test.parquet")
        if "validation" in file:
            new_file = os.path.join(dirname, "validation.parquet")

        os.rename(file, new_file)

    print (f"Done.")

def _parse_for_lmbff(tasks, n_samples=64, random_state=42):
    """
    LMBFF expects a specific format for the datasets. This function will parse
    the datasets. Instead of randomly sampling in the training phase, ADAPET
    uses a fixed set of examples for each task. This means we have to take a
    subset of the training data as training examples. 
    """

    try:
        assert os.listdir(os.path.join("fine_tuners", "lmbff")) != []
    except AssertionError or FileNotFoundError:
        raise ValueError("LM-BFF repository not found.")

    lmbff_location = os.path.join(fs_glue_location, "lmbff")

    try:
        os.mkdir(lmbff_location)
    except FileExistsError:
        print(f"[setup:parse:lmbff] Directory '{lmbff_location}' already exists. If you'd like to re-run this function, please remove the directory first.")

    for task_name, task in tasks.items():
        output_dir = os.path.join(fs_glue_location, "lmbff", task_name)
        os.mkdir(output_dir)

        for split, data in task.items():
            if "train" in split:
                file_location = os.path.join(output_dir, "train.tsv")
                subset = data.sample(n=n_samples, random_state=random_state)
                subset.to_csv(file_location, sep="\t")

            if "test" in split:
                file_location = os.path.join(output_dir, "test.tsv")
                data.to_csv(file_location, sep="\t")

            if "validation" in split:
                file_location = os.path.join(output_dir, "dev.tsv")
                data.to_csv(file_location, sep="\t")

def _parse_for_adapet(tasks, random_state=42):
    """
    ADAPET expects json format for the datasets. This includes the 'question',
    'passage' 'index' and 'label' fields. 
    """
    
    try:
        assert os.listdir(os.path.join("fine_tuners", "adapet")) != []
    except AssertionError or FileNotFoundError:
        raise ValueError("ADAPET repository not found.")

    adapet_location = os.path.join(fs_glue_location, "adapet")

    try:
        os.mkdir(adapet_location)
    except FileExistsError:
        print(f"[setup:parse:adapet] Directory '{adapet_location}' already exists. If you'd like to re-run this function, please remove the directory first.")

    def rearrange_columns(data):
        return data[["index", "question", "passage", "label"]]
                
    for task_name, task in tasks.items():
        output_dir = os.path.join(fs_glue_location, "adapet", task_name)
        os.mkdir(output_dir)

        for split, data in task.items():
            if "train" in split:
                file_location = os.path.join(output_dir, "train.json")
                subset.to_json(file_location, orient="records", lines=True)

            if "validation" in split:
                file_location = os.path.join(output_dir, "unlabeled.json")
                unlabeled = data.drop(columns="label")
                unlabeled.to_json(file_location, orient="records", lines=True)

def _parse_datasets(fine_tuners: list[str]):
    fs_glue = {}

    for task in fs_glue_task_names:
        task_dir = os.path.join(fs_glue_default_location, task)
        files  = os.listdir(task_dir)
        fs_glue[task] = {}

        for file in filter(lambda x: x.endswith(".parquet"), files):
            fs_glue[task][file.split('.')[0]] = pd.read_parquet(os.path.join(task_dir, file))
              
    for fine_tuner in fine_tuners:
        if fine_tuner == "lmbff":
            _parse_for_lmbff(fs_glue)
        if fine_tuner == "adapet":
            _parse_for_adapet(fs_glue)

def download_and_prepare(
    dataset: AvailableDataset,
    fine_tuner: AvailableFineTuner,
    random_state: int=42
):
    dataset = [ dataset ]
    fine_tuner = [ fine_tuner ]

    if "none" in dataset:
        dataset = [ ]
    if "none" in fine_tuner:
        fine_tuner = [ ]

    if "all" in dataset:
        dataset = [ e.value for e in AvailableDataset ]
    if "all" in fine_tuner:
        fine_tuner = [ e.value for e in AvailableFineTuner ]

    print("Downloading datasets...")
    print(f"Datasets: {dataset}")

    if "fs_glue" in dataset:
        _download_fs_glue()

    print("Parsing datasets...") 
    print(f"Fine-tuners: {fine_tuner}")

    _parse_datasets(fine_tuner) 
 
    print("Done.")
        
if __name__ == "__main__":
    import typer

    typer.run(download_and_prepare)