#!/usr/bin/env python

# huggingface
from transformers import PreTrainedTokenizerBase, AutoTokenizer
from datasets import load_dataset, DatasetDict

# external
from tqdm import tqdm
import typer

# built-in
import os
import json
from typing import Optional
import warnings
from transformers import AutoTokenizer
from transformers import AutoTokenizer

glue_tasks = [ "cola", "mrpc", "qqp", "mnli", "qnli", "rte" , "sst2" ]
superglue_tasks = [ "boolq", "cb", "copa" , "wic" ]
fs_glue_tasks = glue_tasks + superglue_tasks

def download(seed: int, tasks: list[str] = fs_glue_tasks, cache_dir: Optional[str] = None):
    """
    Download the FSGLUE data. If the a cache directory is given, the data
    will be retrieved from the cache if it exists. Otherwise, the data will be
    loaded from the huggingface hub. The downloaded data is then cached if the
    cache_dir is given.
    """
    def read_cache(cache_dir, seed, tasks):
        if cache_dir and os.path.exists(f"{cache_dir}/metadata.json"):
            print(f"Using cached data from './{cache_dir}'. (Found following tasks: {os.listdir(cache_dir)})")
            metadata: dict = json.load(open(f"{cache_dir}/metadata.json"))

            if metadata["train_test_split_shuffle_seed"] == seed:
                for task in tasks:
                    if os.path.exists(f"{cache_dir}/{task}"):
                        fs_glue[task] = DatasetDict.load_from_disk(f"{cache_dir}/{task}")

                return fs_glue, metadata
            else:            
                warnings.warn(f"Found cached data, but the seed does not match the current seed. Re-downloading the data.")
        else:
            if cache_dir:
                warnings.warn(f"Directory {cache_dir} not found. Re-downloading the data.")

    def write_cache(cache_dir, fs_glue: dict[str, DatasetDict], metadata: dict):
        if cache_dir:
            for task, dataset in fs_glue.items():
                dataset.save_to_disk(f"{cache_dir}/{task}")

            json.dump(metadata, open(f"{cache_dir}/metadata.json", "w"))
            warnings.warn(f"Saved data to {cache_dir}")

    fs_glue: dict[ str, DatasetDict ] = {}

    if (cache := read_cache(cache_dir, seed, tasks)):
        fs_glue, metadata = cache
        missing_tasks = set(tasks) - set(fs_glue.keys())

        if len(missing_tasks) == 0:
            return fs_glue, metadata

        warnings.warn(f"Found cached data, but the following tasks are missing: {missing_tasks}. Re-downloading the data.")        
        tasks = missing_tasks

    for task in (pbar := tqdm(tasks, desc="Loading FSGLUE tasks")):
        pbar.set_postfix(task=task)
        split = DatasetDict()

        try:
            dataset = load_dataset("glue", task)
        except:
            dataset = load_dataset("super_glue", task)

        # mnli has two versions of the validation and test sets
        # the matched versions do more closely resemble the train set
        # while the mismatched versions are more supposed to be more challenging
        if task == "mnli":
            split["train"] = dataset["train"]
            split["validation"] = dataset["validation_mismatched"]
            split["test"] = dataset["test_mismatched"]
        # some are not split by default
        elif not isinstance(dataset, DatasetDict):
            split: DatasetDict = dataset.train_test_split(
                shuffle=True,
                seed=seed
            )
        # others are already split
        else:
            split = dataset

        fs_glue[task] = split

    metadata: dict = { "train_test_split_shuffle_seed": seed }

    write_cache(cache_dir, seed, fs_glue, metadata)

    return fs_glue, metadata

def process_data_bert(dataset, tokenizer):
    """
    Function to process the data for BERT-like models. This function
    concatenates strings with the [SEP] token and tokenizes the data to prepare
    them for SequenceClassification.
    """
    def concatenate(sentences):
        return "[CLS] " + " [SEP] ".join(sentences) + " [SEP]"

    def get_strings(row):
        return filter(lambda value: isinstance(value, str), row.values())

    def concatenate_sentences(dataset):
        return { "prompt": concatenate(get_strings(dataset)), "label": dataset["label"] }

    def tokenize_function(batch):
        return tokenizer(batch["prompt"], padding="max_length", truncation=True)

    return dataset.map(concatenate_sentences).map(tokenize_function, batched=True)

def prepare(data: dict[str, DatasetDict], metadata: dict, tokenizer: PreTrainedTokenizerBase, cache_dir: Optional[str]):
    """
    Tokenize and cache the processed data. If cache_dir is given, the processed
    data is attempted to be retrieved from the cache. Otherwise, it's processed
    and then cached.
    """

    def read_cache(cache_dir: Optional[str], data: dict[str, DatasetDict], metadata: dict, tokenizer: PreTrainedTokenizerBase):
        if cache_dir:
            if "processed" in metadata and metadata["processed"]["model"] == str(tokenizer.name_or_path):
                print(f"Using cached processed data from './{cache_dir}'. (Found the following metadata: {metadata["processed"]})")
                return data, metadata
            else:
                warnings.warn(f"Found cached data, but the model does not match the current model. Re-processing the data.")

    def write_cache(cache_dir: Optional[str], data: dict[str, DatasetDict], metadata: dict):
        if cache_dir:
            for task, dataset in data.items():
                dataset.save_to_disk(f"{cache_dir}/{task}")

            json.dump(metadata, open(f"{cache_dir}/metadata.json", "w"))
            print(f"Saved processed data to {cache_dir}")

    if (cache := read_cache(cache_dir, data, metadata, tokenizer)):
        return cache

    data = { task: process_data_bert(dataset, tokenizer) for task, dataset in data.items() }
    metadata["processed"] = { "model": tokenizer.name_or_path }

    write_cache(cache_dir, data, metadata)

    return data, metadata

def download_and_prepare(seed: int, tasks: list[str] = fs_glue_tasks, cache_dir: Optional[str] = "data", tokenizer: Optional[PreTrainedTokenizerBase] = None):
    """
    Download and prepare the data for the given tasks. If a tokenizer is not
    given, preparation is skipped. If cache_dir is not given, cache is not used.
    """
    data, metadata = download(seed, tasks, cache_dir)

    if tokenizer:
        data, metadata = prepare(data, metadata, tokenizer, cache_dir)

    return data, metadata
    
def download_and_prepare_cli(seed: int = 42, model_name: str = "bert-base-uncased", tasks: list[str] = fs_glue_tasks, output_dir: str = "data"):
    """
    Download and prepare the FSGLUE data for the given tasks. If a tokenizer
    is not given, preparation is skipped. If cache_dir is not given, cache is
    not used.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    download_and_prepare(seed, tasks, output_dir, tokenizer)

if __name__ == "__main__":
    typer.run(download_and_prepare_cli)