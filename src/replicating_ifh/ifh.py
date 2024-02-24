#!/usr/bin/env python

# huggingface
import time
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification, HfArgumentParser
from datasets import load_dataset, DatasetDict
from evaluate import combine
from peft import get_peft_model, LoraConfig, TaskType

# external
from tqdm import tqdm
import wandb

# built-in
import numpy as np
import os
import json
import dataclasses
from enum import Enum
from typing import Any, Optional

glue_tasks = [ "cola", "mrpc", "qqp", "mnli", "qnli", "rte" , "sst2" ]
superglue_tasks = [ "boolq", "cb", "copa" , "wic" ]
fs_glue_tasks = glue_tasks + superglue_tasks

def get_data(seed: int, tasks=fs_glue_tasks):
    fs_glue: dict[ str, DatasetDict ] = {}

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
            split["validation"] = dataset["validation_mismatched"]
            split["test"] = dataset["test_mismatched"]

        # some are not split by default
        if not isinstance(dataset, DatasetDict):
            split: DatasetDict = dataset.train_test_split(
                shuffle=True,
                seed=seed
            )
        else:
            split = dataset.shuffle(seed=seed)

        fs_glue[task] = split

    return fs_glue

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

    return evaluation

class MethodRunner():
    methods = {
        "lora" : run_lora,
        "adam" : run_adam,
        "baseline" : run_baseline
    }

    @staticmethod
    def run(method: str, *args: Any, **kwargs: Any):
        if method not in MethodRunner.methods:
            raise ValueError(f"Method {method} not found in {MethodRunner.methods.keys()}")

        return MethodRunner.methods[method](*args, **kwargs)

    @staticmethod
    def available_methods():
        return list(MethodRunner.methods.keys())

@dataclasses.dataclass
class CustomArguments:
    """Arguments for the script. If you'd like to change the training
    configuration, use TrainingArguments."""

    train_test_split_shuffle_seed: int = dataclasses.field(
        default=42,
        metadata={ "help": """Seed for the train_test_split shuffle.""" }
    )
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
    use_wandb: bool = dataclasses.field(
        default=False, 
        metadata={ "help": """Whether to use wandb for logging. If True, make
                           you are logged in.""" }
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
        os.environ["WANDB_PROJECT"] = "ifh"
        os.environ["WANDB_LOG_MODEL"] = "false"
        os.environ["WANDB_WATCH"] = "false"

        # weird error if str instead of list of str
        # https://github.com/huggingface/transformers/issues/22429
        training_args.report_to = ["wandb"] 

    # populate list with all methods
    args.methods = args.methods or MethodRunner.available_methods()

    # load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)

    # load the data
    data = get_data(training_args.seed, args.tasks or fs_glue_tasks)

    # process the data
    data = { task: process_data_bert(dataset, tokenizer) for task, dataset in data.items() }
    top_level_output_dir = training_args.output_dir

    for method in args.methods + ["baseline"]:
        for task, dataset in data.items():
            training_args.output_dir = f"{top_level_output_dir}/{args.model_name}/{method}/{task}" 

            train_set = dataset["train"].select(range(64))
            eval_set = dataset["validation"]

            wandb.init(project="ifh", name=training_args.output_dir, reinit=True)
            results = MethodRunner.run(method, training_args, model, train_set, eval_set)
            wandb.log(results)
            wandb.finish()

if __name__=="__main__":
    parser = HfArgumentParser([CustomArguments, TrainingArgumentsCustomDefaults])

    parser.add_argument("--list-tasks", action="store_true", help="List all available tasks.")
    parser.add_argument("--list-methods", action="store_true", help="List all avaible methods.")
    args = parser.parse_args_into_dataclasses()

    if args[-1].list_tasks:
        print(fs_glue_tasks)
        exit(0)

    if args[-1].list_methods:
        print(MethodRunner.available_methods())
        exit(0)

    run(args)