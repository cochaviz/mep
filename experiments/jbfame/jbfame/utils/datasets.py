import os
import random
import shutil
import subprocess
from typing import Optional

import pandas as pd
from regex import W
import seaborn as sns
from torch import Value
from tqdm import tqdm
from datasets import DatasetDict
from transformers import PreTrainedTokenizerBase

from jbfame import data, utils
from jbfame.utils.params import ExperimentArguments

import logging
logger = logging.getLogger(__name__)

def load(
    args: ExperimentArguments,
    restrict_sampled: bool = True,
    try_preprocessed_from_remote = True,
) -> DatasetDict:
    """
    Reads *.parquet files in data_dir and returns them as a
    DatasetDict where the key is the file name (without the extension) and the
    value the dataset.
    """

    def insert_unsafe_column(datasets: DatasetDict):
        """
        Assigns the 'unsafe' column from the null dataset to all other datasets
        based on the question ID.
        """
        assert "null" in datasets and "unsafe" in datasets["null"].column_names, "The null task has to be present and tagged in order to insert the unsafe column."

        logger.info("Inserting 'unsafe' column in all datasets based on the null task.")

        try:
            for task, dataset in datasets.items():
                if "q_id" not in dataset.column_names and task != "null":
                    raise ValueError(f"q_id column not found in task {task}.")
                if task == "null" or ("unsafe" in dataset.column_names and "chat" in dataset.column_names):
                    logger.info(f"Skipping task {task}; already has 'unsafe' column.")
                    continue

                datasets[task] = dataset.map(lambda row: { 
                        **row, 
                        "unsafe": datasets["null"][row["q_id"]]["unsafe"], 
                        "chat":  [{ 
                            "role": "user", 
                            "content": row["prompt"] 
                        }]
                    })
        except:
            raise ValueError("It appears that the question safety index does not have the same shape as the null dataset. Please make sure the question safety index is based on the current 'null' task.") 

        return datasets

    def sample_questions(datasets: DatasetDict, sample_size: int, restrict: bool):
        """
        Selects a subset of the questions in the null dataset and filters the
        other tasks based on the question ID. Ensures that only the same
        questions are sampled. If restrict is set, the other tasks are also
        limited to 'sample_size' samples.
        """
        indices = range(len(datasets["null"]))
        logger.info(f"Sampling {sample_size} questions from all tasks.")

        try:
            indices = random.sample(indices, k=sample_size)
            datasets["null"] = datasets["null"].select(indices)
        except ValueError:
            logger.warning("Sample size exceeds the number of questions in the null task. Returning all questions.")

        for task, dataset in datasets.items():
            if "q_id" not in dataset.column_names and task != "null":
                raise AssertionError(f"q_id column not found in task {task}.")
            if task == "null":
                logger.info(f"Skipping task {task}.")
                continue

            datasets[task] = dataset.filter(lambda row: row["q_id"] in indices)

            if restrict:
                try:
                    sampled_indices_task = random.sample(range(len(datasets[task])), sample_size)
                    datasets[task] = datasets[task].select(sampled_indices_task)
                except ValueError:
                    logger.warning(f"Sample size exceeds the number of questions in the {task} task. Returning all questions.")

        return datasets

    def load_preprocessed_from_remote(tasks: list[str]):
        """
        Loads the preprocessed datasets from the remote location. If the
        datasets are not found, they are downloaded and prepared.
        """
        remote_path = "https://github.com/cochaviz/mep/raw/experiments/experiments/jbfame/assets/data_preprocessed"
        os.makedirs(args.data_dir, exist_ok=False)

        if "null" not in tasks:
            tasks = ["null"] + tasks

        logger.info(f"Downloading preprocessed datasets from remote (tasks: {tasks}).")

        
        for task in (pbar := tqdm(
            tasks, 
            desc="Downloading preprocessed datasets from remote", 
            disable=args.disable_tqdm
        )):
            pbar.set_postfix_str(f"Downloading {task}")

            remote_task_path = f"{remote_path}/{args.model_path}/{task}.parquet"
            local_task_path = f"{args.data_dir}/{task}.parquet"

            subprocess.run(
                f"wget -qO {local_task_path} {remote_task_path}",
                shell=True, check=True
            )

    tasks = args.tasks or data.available_tasks()

    # if the data directory does not exist, download the data
    if not try_preprocessed_from_remote or os.path.exists(args.data_dir):
        logger.info(f"Data directory ({args.data_dir}) exists. Loading datasets from local storage.")
        datasets = data.download_and_prepare(tasks, args.data_dir, args.disable_tqdm) 
    else:
        try:
            load_preprocessed_from_remote(tasks)
        except subprocess.CalledProcessError:
            shutil.rmtree(args.data_dir)
            logger.warning("Failed to download (some) preprocessed datasets from remote, or not all required tasks were available. Downloading and preparing datasets locally.")
        finally:
            datasets = data.download_and_prepare(tasks, args.data_dir, args.disable_tqdm)

    # always the case when downloading with the 'download_and_prepare' function
    assert "null" in datasets, "The null task has to be present in the dataset."

    # if the null task is present, insert the unsafe column in all other tasks
    if "unsafe" in datasets["null"].column_names:
        datasets = insert_unsafe_column(datasets) 

    # if the task size is set, sample the datasets
    if args._task_size:
        sample_questions(datasets, args._task_size, restrict_sampled)

    return datasets

def preprocess(
    datasets: DatasetDict,
    args: ExperimentArguments,
    tokenizer: Optional[PreTrainedTokenizerBase] = None, # skips tokenization if None
    remove_unused_columns: bool = True,
) -> DatasetDict:
    """
    Takes a dataset and preprocesses it by adding the response to the prompt,
    and tokenizing the prompt and response. If tokenizer is None, only the
    response is added to all the prompts in the huggingface 'chat' format which
    can be found in the 'chat' column. If character_limit is not None, all
    prompts exceeding this limit are removed. 
    """
    def set_expected_response(
        row: dict, 
        response_safe: str = args.response_safe, 
        response_unsafe: str = args.response_unsafe
    ):
        # NOTE: the 'chat' column is overriden here
        row["chat"] = [
            { 
                "role": "user", 
                "content": row["prompt"] 
            }, { 
                "role": "assistant", 
                "content": response_unsafe if row["unsafe"] else response_safe
            }  
        ]
        return row    

    def tokenize(batch: dict):
        assert tokenizer, "Tokenizer has to be provided in order to tokenize the data."

        batch["chat"] = tokenizer.apply_chat_template(
            batch["chat"], 
            return_tensors="pt", 
            padding=True
        ).to(utils.models.DEVICE) # type: ignore

        return batch

    logger.info("Preprocessing datasets.")
    logger.debug("Tasks: " + ", ".join(datasets.keys()))

    # we generally only care about the unsafe questions, since otherwise we
    # cannot determine whether it is a successful jailbreak prompt
    processed = _unsafe_questions(datasets, args)

    if args.max_prompt_length:
        logger.info(f"Filtering prompts longer than {args.max_prompt_length} tokens.")

        # HACK: ideally, the token limit would be based on the actual number of
        # tokens. Here, it is based on the number of characters in the prompt. This
        # is not ideal, but it allows for easy batching and is good enough for me.
        processed = _filter_prompt_length(processed, args.max_prompt_length)

    # filtering jailbreaks only after the prompt length has been filtered, since
    # it saves a lot of time
    processed = _jailbreak_prompts(processed, args)

    if not tokenizer:
        return processed

    # this will filter out False and None values for the 'jailbreak' column
    jailbroken = processed.filter(lambda row: row["jailbreak"])

    if jailbroken.num_rows == 0:
        raise ValueError("No jailbreak prompts found. Please make sure the 'jailbreak' column is present in the dataset and some prompts incite jailbreaks.")

    logger.info(f"Jailbreak prompts: {jailbroken.num_rows}.")
    logger.info("Tokenizing prompts and responses.")

    # expected responses are only added after since, if we do not have a
    # tokenizer, we do not expect to run the full experiment
    processed = processed.map(
        set_expected_response,
        batched=True
    )

    #  all columns except the input_ids, since we only need this for training
    #  and inference
    all_but_input_ids: list[str] = list(set(datasets["null"].column_names) - set(["input_ids", "q_id"]))
    logger.debug("Removing columns: " + ", ".join(all_but_input_ids))

    # because the 'apply chat template' does not work in batch mode, we have to
    # do this separately
    return processed.map(tokenize).map(
        tokenize,
        remove_columns=all_but_input_ids if remove_unused_columns else None,
        batched=True
    )

def persist(
    args: ExperimentArguments, 
    datasets: Optional[DatasetDict], 
    save_dir: str = "assets/data_preprocessed",
    tag: bool = True,
):
    def write_to_file(datasets: DatasetDict):

        output_path = f"{save_dir}/{args.model_path}" if tag else save_dir

        os.makedirs(output_path, exist_ok=False)

        for task, dataset in datasets.items():
            dataset.to_parquet(f"{output_path}/{task}.parquet")
        
        return output_path

    if not datasets:
        datasets = load(args)
    
    return write_to_file(datasets)

def info(
    args: ExperimentArguments, 
    parameter: Optional[str] = "length",
    figure_path: Optional[str] = None, 
    datasets: Optional[DatasetDict] = None,
) -> pd.DataFrame:
    """
    Returns a description of each task in the dataset, and creates a histogram
    of the length of the prompts for each task. If figure_path is not None, the
    plot is saved to this location. Otherwise, call plt.show() to display the
    plot.
    """

    def set_length(row):
        row["length"] = len(row["prompt"])
        return row 

    if not parameter == "length":
        raise NotImplementedError("Only parameter 'length' can be used...")

    if not datasets: 
        datasets = load(args)
        datasets = preprocess(
            datasets,
            args,
        )

    param_df = pd.DataFrame()

    for task, dataset in datasets.items():
        if parameter == "length":
            param_set = dataset.map(set_length, remove_columns=dataset.column_names)

        param_set = param_set.add_column("task", [task] * len(dataset))
        param_df = pd.concat([param_df, param_set.to_pandas()])

    sns.histplot(
        data=param_df, 
        x=parameter, 
        hue="task", 
        kde=True, 
        common_norm=False, 
        binwidth=200, 
        edgecolor="black"
    )

    if figure_path:
        import matplotlib.pyplot as plt
        sns.set_theme(context="paper")
        plt.savefig(figure_path)

    return param_df.groupby("task").describe()

# def _set_chat(datasets: DatasetDict, column:str) -> DatasetDict:
#     def set_chat(row):
#         row["chat"] = [{ "role": "user", "content": row["prompt"] }]
#         return row

#     for task, dataset in datasets.items():
#         if column in dataset.column_names:
#             continue

#         datasets[task] = dataset.map(set_chat)

#     return datasets

def _all_have_column(datasets: DatasetDict, column: str) -> bool:
    return all(map(lambda dataset: column in dataset.column_names, datasets.values()))

def _unsafe_questions(
    datasets: DatasetDict,
    args: ExperimentArguments,
) -> DatasetDict:
    # if unsafe column not present in null, generate it
    if "unsafe" not in datasets["null"].column_names:
        # we only care about the 'null' dataset when it comes to safe or unsafe
        # questions, since all other tasks depend on null for the actual questions
        datasets["null"] = utils.safety.tag_prompt_safety(
            datasets["null"], args # type: ignore
        )
        datasets["null"].to_parquet(os.path.join(args.data_dir, "null.parquet"))
    else:
        logger.info("Unsafe column already present in null task. Not generating.")

    if "unsafe" in datasets["null"].column_names and not _all_have_column(datasets, "unsafe"):
        # on reload, the unsafe column is inserted in all datasets 
        datasets = utils.datasets.load(args)
    else:
        logger.info("Unsafe column already present in all tasks. Not generating.")

    return datasets

def _jailbreak_prompts(
    datasets: DatasetDict,
    args: ExperimentArguments,
) -> DatasetDict:
    if not _all_have_column(datasets, "jailbreak"):
        # generate the jailbreak column, i.e. respond to the prompts and check
        # whether the conversation is unsafe
        datasets = utils.safety.tag_prompt_jailbreak(
            datasets, args 
        )
    else:
        logger.info("Jailbreak column already present in all tasks. Not generating.")

    return datasets

def _filter_prompt_length(
    datasets: DatasetDict,
    max_prompt_length: int,
) -> DatasetDict:
    def length_less_than(
        row, 
    ) -> bool:
        return len(row["prompt"]) <= max_prompt_length

    filtered = datasets.filter(length_less_than)
    empty = [ task if dataset.num_rows == 0 else None for task, dataset in filtered.items() ]
    empty = list(filter(lambda task: task is not None, empty))

    if len(empty)> 0:
        raise ValueError(f"No questions left in tasks {empty} after filtering by length of {max_prompt_length}.")
    for task, dataset in filtered.items():
        logger.info(f"Task {task}: {dataset.num_rows} questions left after filtering by length of {max_prompt_length}.")

    return filtered


def __main__():
    other_args = [
        ("--stage", dict(type=str, default="all", help="Stage to run. Options: tag_safety, tag_jailbreak, all")),
    ]
    experiment_args, other_args = utils.params.parse_default_args(other_args)
    datasets = load(experiment_args)

    if other_args.stage == "tag_safety":
        datasets = _unsafe_questions(datasets, experiment_args)
    if other_args.stage == "tag_jailbreak":
        datasets = _jailbreak_prompts(datasets, experiment_args)
    if other_args.stage == "all":
        datasets = preprocess(datasets, experiment_args)

    for task, dataset in datasets.items():
        dataset.to_parquet(f"{experiment_args.data_dir}/{task}.parquet") # type: ignore

