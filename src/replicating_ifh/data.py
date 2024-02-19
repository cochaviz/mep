#!/usr/bin/env python3

import glob
import os
from enum import Enum
import subprocess

import pandas as pd
from datasets import load_dataset_builder

from fine_tuners import AvailableFineTuner

fs_glue_location = os.path.join(os.getcwd(), "data/fs_glue")
fs_glue_default_location = os.path.join(fs_glue_location, "default")

glue_task_names = [ "cola", "mrpc", "qqp", "mnli", "qnli", "rte" , "sst2" ]
superglue_task_names = [ "boolq", "cb", "copa" , "wic" ]
fs_glue_task_names = glue_task_names + superglue_task_names

class DatasetOptions(str, Enum):
    none = "none"
    all = "all"

class AvailableDataset(str, Enum):
    fs_glue = "fs_glue"
    jfmd = "jfmd"

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
        new_file = ""

        if "train" in file:
            new_file = os.path.join(dirname, "train.parquet")
        if "test" in file:
            new_file = os.path.join(dirname, "test.parquet")
        if "validation" in file:
            new_file = os.path.join(dirname, "validation.parquet")

        os.rename(file, new_file)

    print (f"Done.")

def _download_jfmd():
    print(subprocess.call([ "bash", "jfmd.sh" ], cwd="data"))

def _prepare_for_lmbff(tasks, n_samples=64, random_state=42, force=False):
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

    if force:
        import shutil
        shutil.rmtree(lmbff_location)

    try:
        os.mkdir(lmbff_location)
    except FileExistsError:
        print(f"[setup:parse:lmbff] Directory '{lmbff_location}' already exists. If you'd like to re-run this function, please remove the directory first.")
        return

    for task_name, task in tasks.items():
        output_dir = os.path.join(fs_glue_location, "lmbff", task_name)
        os.mkdir(output_dir)

        for split, data in task.items():
            if "train" in split:
                file_location = os.path.join(output_dir, "train.tsv")
                subset = data.sample(n=n_samples, random_state=random_state)
                subset.to_csv(file_location, sep="\t", index=False)

            if "test" in split:
                file_location = os.path.join(output_dir, "test.tsv")
                data.to_csv(file_location, sep="\t", index=False)

            if "validation" in split:
                file_location = os.path.join(output_dir, "dev.tsv")
                data.to_csv(file_location, sep="\t", index=False)

def _prepare_for_adapet(tasks, random_state=42):
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
        return

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

def _download(datasets: list[AvailableDataset]):
    print("Downloading datasets...")
    print(f"Datasets: {datasets}")

    for dataset in datasets:
        if dataset == AvailableDataset.jfmd:
            _download_jfmd()
        if dataset == AvailableDataset.fs_glue:
            _download_fs_glue()

def _prepare(fine_tuners: list[str], force=False):
    print("Parsing datasets...") 
    print(f"Fine-tuners: {fine_tuners}")
    
    if force:
        print("Forcing re-parsing of datasets.")

    fs_glue = {}

    for task in fs_glue_task_names:
        task_dir = os.path.join(fs_glue_default_location, task)
        files  = os.listdir(task_dir)
        fs_glue[task] = {}

        for file in filter(lambda x: x.endswith(".parquet"), files):
            fs_glue[task][file.split('.')[0]] = pd.read_parquet(os.path.join(task_dir, file))
              
    for fine_tuner in fine_tuners:
        if fine_tuner == AvailableFineTuner.lmbff:
            _prepare_for_lmbff(fs_glue, force=force)
        if fine_tuner == AvailableFineTuner.adapet:
            _prepare_for_adapet(fs_glue)

def download_and_prepare(
    dataset: DatasetOptions | AvailableDataset,
    fine_tuner: AvailableFineTuner,
    random_state: int=42,
    force: bool=False
):
    datasets = [ dataset ]
    fine_tuners = [ fine_tuner ]

    if dataset == DatasetOptions.none:
        datasets = [ ]
    if fine_tuner == AvailableFineTuner.none:
        fine_tuners = [ ]

    if dataset == DatasetOptions.all:
        datasets = list(AvailableDataset)
    if fine_tuner == AvailableFineTuner.all:
        fine_tuners = list(AvailableFineTuner)

    _download(datasets)
    _prepare(fine_tuners, force=force) 
 
    print("Done.")
        
if __name__ == "__main__":
    import typer
    typer.run(download_and_prepare)