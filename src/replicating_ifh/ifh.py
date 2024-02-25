#!/usr/bin/env python

# huggingface
import time
from transformers import Trainer, TrainingArguments, AutoTokenizer, PreTrainedTokenizerBase, AutoModelForSequenceClassification, HfArgumentParser
from datasets import load_dataset, DatasetDict
from evaluate import combine
from peft import get_peft_model, LoraConfig, TaskType

# external
from tqdm import tqdm

# built-in
import numpy as np
import os
import json
import dataclasses
from typing import Any, Optional
import warnings

glue_tasks = [ "cola", "mrpc", "qqp", "mnli", "qnli", "rte" , "sst2" ]
superglue_tasks = [ "boolq", "cb", "copa" , "wic" ]
fs_glue_tasks = glue_tasks + superglue_tasks

def get_data(seed: int, tasks: list[str] = fs_glue_tasks, cache_dir: Optional[str] = None):
    def read_cache(cache_dir, seed, tasks):
        if cache_dir and os.path.exists(f"{cache_dir}/metadata.json"):
            print(f"Using cached data from {cache_dir}. Set cache_dir=None to re-download the data.")
            metadata = json.load(open(f"{cache_dir}/metadata.json"))

            if metadata["train_test_split_shuffle_seed"] == seed:
                for task in tasks:
                    if os.path.exists(f"{cache_dir}/{task}"):
                        fs_glue[task] = DatasetDict.load_from_disk(f"{cache_dir}/{task}")

                return fs_glue, metadata
            else:            
                warnings.warn(f"Found cached data, but the seed does not match the current seed. Re-downloading the data.")
        else:
            if cache_dir:
                warnings.warn(f"Cache directory {cache_dir} not found. Re-downloading the data.")

    def write_cache(cache_dir, seed, fs_glue: dict[str, DatasetDict], metadata: dict[str, Any]):
        if cache_dir:
            for task, dataset in fs_glue.items():
                dataset.save_to_disk(f"{cache_dir}/{task}")

            metadata = { "train_test_split_shuffle_seed": seed }
            json.dump(metadata, open(f"{cache_dir}/metadata.json", "w"))
            warnings.warn(f"Saved data to {cache_dir}")

    fs_glue: dict[ str, DatasetDict ] = {}

    if (fs_glue, metadata := read_cache(cache_dir, seed, tasks)):
        missing_tasks = set(tasks) - set(fs_glue.keys())

        if len(missing_tasks) == 0:
            return fs_glue, metadata

        print(f"Found cached data, but the following tasks are missing: {missing_tasks}. Re-downloading the data.")        
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

    metadata = { "train_test_split_shuffle_seed": seed }

    write_cache(cache_dir, seed, fs_glue, metadata)

    return fs_glue, metadata

def process_data_bert(dataset, tokenizer):
    def concatenate(sentences):
        return "[CLS] " + " [SEP] ".join(sentences) + " [SEP]"

    def get_strings(row):
        return filter(lambda value: isinstance(value, str), row.values())

    def concatenate_sentences(dataset):
        return { "prompt": concatenate(get_strings(dataset)), "label": dataset["label"] }

    def tokenize_function(batch):
        return tokenizer(batch["prompt"], padding="max_length", truncation=True)

    return dataset.map(concatenate_sentences).map(tokenize_function, batched=True)

def process_data(data: dict[str, DatasetDict], metadata: Optional[dict], tokenizer: PreTrainedTokenizerBase, cache_dir: Optional[str]):
    def read_cache(cache_dir: Optional[str], data: dict[str, DatasetDict], metadata: dict, tokenizer: PreTrainedTokenizerBase):
        if cache_dir and metadata:
            print(f"Using cached processed data from {cache_dir}. Set cache_dir=None to re-process the data.")

            if "processed" in metadata and metadata["processed"]["model"] == str(tokenizer.name_or_path):
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


def baseline_majority(train_set, eval_set, metrics=["accuracy"]):
    train_set.with_format("numpy")

    # get majority class
    majority_class_index = np.argmax(np.bincount(train_set["label"]))
    majority_class = train_set["label"][majority_class_index]

    train_set.reset_format()

    metric = combine(metrics)
    return metric.compute(
        predictions=[majority_class] * len(eval_set["label"]), 
        references=eval_set["label"])

def compute_metrics(eval_pred, metrics=["accuracy"]):
    metric = combine(metrics)
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def run_adam(args: TrainingArguments, model, train_set, eval_set):
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        compute_metrics=compute_metrics
    )
    return trainer.train()

def run_lora(args: TrainingArguments, model, train_set, eval_set):
    config = LoraConfig(
        task_type= TaskType.SEQ_CLS, 
        lora_alpha=32, 
        lora_dropout=0.1
    )
    trainer = Trainer(
        model= get_peft_model(model, config),
        args=args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        compute_metrics=compute_metrics
    )
    return trainer.train()

def run_baseline(args: TrainingArguments, _, train_set, eval_set):
    evaluation = baseline_majority(train_set, eval_set)  

    if not args.report_to:
        os.mkdir(f"{args.output_dir}")
        open(f"{args.output_dir}/evaluation.json", "w").write(json.dumps(evaluation))
    if args.use_wandb:
        import wandb
        wandb.log(evaluation)
        wandb.finish()

    return evaluation

class Methods():
    _method_run_map = {
        "lora" : run_lora,
        "adam" : run_adam,
        "baseline" : run_baseline
    }

    @staticmethod
    def run(method: str, *args: Any, **kwargs: Any) -> dict:
        if method not in Methods._method_run_map:
            raise ValueError(f"Method {method} not found in {Methods._method_run_map.keys()}")

        return Methods._method_run_map[method](*args, **kwargs)

    @staticmethod
    def list_available():
        return list(Methods._method_run_map.keys())

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
    train_sample_seed: int = dataclasses.field(
        default=42,
        metadata={ "help": """Seed for sampling 'train_set_size' examples from
                           the training set.""" }
    )
    train_set_size: int = dataclasses.field(
        default=64,
        metadata={ "help": """Number of examples used to represent each
                           class in the few-shot learning regime.""" }
    )
    use_wandb: bool = dataclasses.field(
        default=False, 
        metadata={ "help": """Whether to use wandb for logging. If True, make
                           you are logged in.""" }
    )
    cache_dir: Optional[str] = dataclasses.field(
        default="data",
        metadata={ "help": """Path to the dataset cache directory. If None, no
                           caching will be used.""" }
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

def run(args: CustomArguments, training_args: TrainingArgumentsCustomDefaults):
    if args.use_wandb:
        import wandb

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
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)

    # load and process the data
    if args.use_wandb:
        wandb.init(
            project="ifh", 
            name=f"{training_args.output_dir}/pre_processing", 
            tags=[training_args.output_dir, args.model_name, "pre_processing"],
            reinit=True
        )

    data, metadata = get_data(args.train_test_split_shuffle_seed, args.tasks or fs_glue_tasks, args.cache_dir)
    data, metadata = process_data(data, metadata, tokenizer, args.cache_dir)

    if args.use_wandb:
        wandb.finish()

    top_level_output_dir = training_args.output_dir

    for method in set(args.methods + ["baseline"]):
        for task, dataset in data.items():
            training_args.output_dir = f"{top_level_output_dir}/{args.model_name}/{method}/{task}" 

            train_set = dataset["train"].shuffle(seed=args.train_sample_seed).select(range(args.train_set_size))
            eval_set = dataset["validation"].select(min(len(dataset["validation"]), range(2000)))

            if args.use_wandb:
                wandb.init(
                    project="ifh", 
                    name=training_args.output_dir, 
                    tags=[top_level_output_dir, args.model_name, method, task],
                    reinit=True
                )

            Methods.run(method, training_args, model, train_set, eval_set)

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

    run(args)