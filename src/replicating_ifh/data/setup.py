#!/usr/bin/env python3

import glob
import os
import pandas as pd

from datasets import load_dataset_builder

fs_glue_default_location = os.getcwd() + "/data/fs_glue/default"

glue_task_names = [ "cola", "mrpc", "qqp", "mnli", "qnli", "rte" , "sst2" ]
superglue_task_names = [ "boolq", "cb", "copa" , "wic" ]
fs_glue_task_names = glue_task_names + superglue_task_names

def download_datasets():
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

def parse_for_lmbff(tasks):
    """
    LMBFF expects a specific format for the datasets. This function will parse
    the datasets. All scripts in the LM-BFF repository are required to do this.
    Make sure that the fine-tuners are set up before running this function.
    """

    try:
        assert os.listdir("lmbff") != []
    except AssertionError or FileNotFoundError:
        raise ValueError("LM-BFF repository not found.")

    print(tasks)

def parse_for_adapet(tasks):
    """
    LMBFF expects a specific format for the datasets. This function will parse
    the datasets. All scripts in the LM-BFF repository are required to do this.
    Make sure that the fine-tuners are set up before running this function.
    """

    try:
        assert os.listdir("adapet") != []
    except AssertionError or FileNotFoundError:
        raise ValueError("ADAPET repository not found.")

    print(tasks)

def parse_datasets():
    fs_glue = {}

    # convert glue tasks to various formats
    for dir in os.walk(fs_glue_default_location):
        for file in dir[2]:
            if file.endswith(".parquet"):
                filepath = dir[0] + "/" + file
                fs_glue[file] = pd.read_parquet(filepath)

        break
              
    # parse_for_lmbff(fs_glue)  
    parse_for_adapet(fs_glue)
        
if __name__ == "__main__":
    download_datasets()
    # parse_datasets()
