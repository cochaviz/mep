#!/usr/bin/env python

# huggingface
import time
from transformers import TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification, HfArgumentParser
from datasets import Dataset

# external
from pyarrow import Table
import pandas as pd
try:
    import wandb
except ImportError:
    warnings.warn("wandb not installed. Run `pip install wandb` to use wandb.")

# internal
from data import download_and_prepare, fs_glue_tasks
from methods import Methods

# built-in
import os
import dataclasses
from typing import Optional
import warnings

def extract_few_shot(dataset: pd.DataFrame(), num_samples: int, sample_seed: int) -> pd.DataFrame:
    few_shot = pd.DataFrame()
    labels = dataset["label"].unique()

    for label in labels:
        few_shot = pd.concat([
            few_shot,
            dataset[dataset["label"] == label].sample(num_samples, random_state=sample_seed)
        ])

    return few_shot, len(labels)

@dataclasses.dataclass
class CustomArguments:
    """Arguments for the script. If you'd like to change the training
    configuration, use TrainingArguments."""

    model_name: str = dataclasses.field(
        default="bert-base-uncased", 
        metadata={ "help": """Name of the model, or path as found in the
                           Huggingface model hub.""" }
    )
    tasks: Optional[list[str]] = dataclasses.field(
        default=None, 
        metadata={ "help": """List of tasks to run. If None, all tasks will be
                           run.  To list all available tasks, use
                           --list-tasks.""" }
    )
    methods: Optional[list[str]] = dataclasses.field(
        default=None, 
        metadata={ "help": """List of methods to run. If None, all methods will
                           be run.  To list all available methods, use
                           --list-methods.""" }
    )
    train_test_split_shuffle_seed: int = dataclasses.field(
        default=42,
        metadata={ "help": """Seed for the train_test_split shuffle.""" }
    )
    few_shot_sample_seed: int = dataclasses.field(
        default=42,
        metadata={ "help": """Seed for sampling 'train_set_size' examples from
                           the training set.""" }
    )
    few_shot_samples: int = dataclasses.field(
        default=64,
        metadata={ "help": """Number of examples used to represent each
                           class in the few-shot learning regime.""" }
    )
    use_wandb: bool = dataclasses.field(
        default=False, 
        metadata={ "help": """Whether to use wandb for logging. If True, make
                           you are logged in.""" }
    )
    data_dir: Optional[str] = dataclasses.field(
        default="data",
        metadata={ "help": """Path to the dataset directory. If None, no caching
                           will be used and data will be downloaded on every
                           run.""" }
    )

@dataclasses.dataclass
class TrainingArgumentsCustomDefaults(TrainingArguments):
    output_dir: str = dataclasses.field(
        default=time.strftime("%Y-%m-%d-%H-%M-%S"),
        # metadata=TrainingArguments("").log_level.metadata
    )
    log_level: str = dataclasses.field(
        default="error",
        # metadata=TrainingArguments("").log_level.metadata
    )
    learning_rate: float = dataclasses.field(
        default=1e-5,
        # metadata=TrainingArguments("").learning_rate.metadata
    )
    evaluation_strategy: str = dataclasses.field(
        default="steps",
        # metadata=TrainingArguments("").evaluation_strategy.metadata
    )
    eval_steps: int = dataclasses.field(
        default=8,
        # metadata=TrainingArguments("").eval_steps.metadata
    )
    num_train_epochs: int = dataclasses.field(
        default=1,
        # metadata=TrainingArguments("").num_train_epochs.metadata
    )
    save_strategy: str = dataclasses.field(
        default="no",
        # metadata=TrainingArguments("").save_strategy.metadata
    )

def run( 
        args: CustomArguments,
        training_args: TrainingArgumentsCustomDefaults,
    ):
    if args.use_wandb: 
        os.environ["WANDB_PROJECT"] = "ifh"
        os.environ["WANDB_LOG_MODEL"] = "false"
        os.environ["WANDB_WATCH"] = "false"

        # weird error if str instead of list of str
        # https://github.com/huggingface/transformers/issues/22429
        training_args.report_to = ["wandb"] 

    # populate list with all methods
    args.methods = args.methods or Methods.list_available()

    # load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # load and process the data
    if args.use_wandb:
        wandb.init(
            project="ifh", 
            group="pre_processing",
            name=f"{training_args.output_dir}/pre_processing", 
            tags=[training_args.output_dir, args.model_name, "pre_processing"],
            reinit=True
        )
    data, _ = download_and_prepare(
        args.train_test_split_shuffle_seed, 
        args.tasks,
        args.data_dir, 
        tokenizer
    )
    if args.use_wandb:
        wandb.finish()

    top_level_output_dir = training_args.output_dir

    for method in set(args.methods):
        for task, dataset in data.items():
            training_args.output_dir = f"{top_level_output_dir}/{args.model_name}/{method}/{task}" 

            # select subset of training data and represent
            # each label by train_set_size examples
            dataset["train"].set_format("pandas")
            few_shot_set, num_labels = extract_few_shot(
                dataset["train"], 
                args.few_shot_samples, 
                args.few_shot_sample_seed
            )
            train_set = Dataset(Table.from_pandas(few_shot_set))
            dataset["train"].reset_format()


            # HACK: select subset of validation data because
            # otherwise it takes too long. Should be removed
            # for final run.
            eval_set = dataset["validation"].select(
                range(min(
                    len(dataset["validation"]), 
                    2000
                )))

            if args.use_wandb:
                wandb.init(
                    project="ifh", 
                    name=training_args.output_dir, 
                    group=f"{args.model_name}/{method}",
                    tags=[top_level_output_dir, args.model_name, method, task],
                    reinit=True
                )

            # since number of labels can change between tasks,
            # we need to re-instantiate the model for each task
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model_name, num_labels=num_labels
            )

            try:
                Methods.run(method, training_args, model, train_set, eval_set)
            except Exception as e:
                warnings.warn(f"Error running method {method}: {e}")

                if args.use_wandb:
                    wandb.finish(exit_code=1)
            
            if args.use_wandb:
                wandb.finish()

            # explicitly remove old model
            del model

if __name__=="__main__":
    parser = HfArgumentParser([CustomArguments, TrainingArgumentsCustomDefaults])
    parser.add_argument("--list-tasks", action="store_true", help="List all available tasks.")
    parser.add_argument("--list-methods", action="store_true", help="List all avaible methods.")

    args = parser.parse_args_into_dataclasses()

    if args[-1].list_tasks:
        print(fs_glue_tasks)
        exit(0)

    if args[-1].list_methods:
        print(Methods.list_available())
        exit(0)

    run(*args[:2])