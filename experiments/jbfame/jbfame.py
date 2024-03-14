#!/usr/bin/env python

import os
import warnings
from glob import glob
from typing import Optional
from dataclasses import dataclass, field
import time
from numpy import pad

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, HfArgumentParser, BitsAndBytesConfig, DataCollatorForLanguageModeling, PreTrainedTokenizerBase
from datasets import Dataset, DatasetDict
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from pyarrow import parquet
import seaborn as sns
import pandas as pd
import torch

import data

@dataclass
class TrainingArgumentsCustomDefaults(TrainingArguments):
    output_dir: str = field(
        default=time.strftime("%Y-%m-%d-%H-%M-%S"),
    )
    warmup_steps: int = field(
        default=1,
    )
    per_device_train_batch_size: int = field(
        default=2,
    )
    gradient_accumulation_steps: int = field(
        default=1,
    )
    gradient_checkpointing: bool = field(
        default=True,
    )
    max_steps: int = field(
        default=500,
    )
    learning_rate: float = field(
        default=2.5e-5,
    )
    optim: str = field(
        default="paged_adamw_8bit",
    )


@dataclass
class CustomArguments:
    """Arguments for the script. If you'd like to change the training
    configuration, use TrainingArguments."""

    model_path: str = field(
        default="meta-llama/Llama-2-7b-chat-hf", 
        metadata={ "help": """Name of the model, or path as found in the
                           Huggingface model hub.""" }
    )
    tasks: Optional[list[str]] = field(
        default=None, 
        metadata={ "help": """List of tasks to run. If None, all tasks will be
                           run.  To list all available tasks, use
                           --list-tasks.""" }
    )
    unsafe_response: str = field(
        default="Sorry, but I cannot help with that.",
        metadata={ "help" : """Response the model should give to unsafe
                            questions.""" }
    )
    max_prompt_length: int = field(
        default=512,
        metadata={ "help": """Maximum length of the prompt (in number of
                           tokens). Any datapoints over this amount are
                           discarded.""" }
    )
    # methods: Optional[list[str]] = field(
    #     default=None, 
    #     metadata={ "help": """List of methods to run. If None, all methods will
    #                        be run.  To list all available methods, use
    #                        --list-methods.""" }
    # )
    # train_test_split_shuffle_seed: int = field(
    #     default=42,
    #     metadata={ "help": """Seed for the train_test_split shuffle.""" }
    # )
    shuffle: Optional[int] = field(
        default=None,
        metadata={ "help": """Seed for sampling 'train_set_size' examples from
                           the training set.""" }
    )
    # few_shot_samples: int = field(
    #     default=64,
    #     metadata={ "help": """Number of examples used to represent each
    #                        class in the few-shot learning regime.""" }
    # )
    use_wandb: bool = field(
        default=False, 
        metadata={ "help": """Whether to use wandb for logging. If True, make
                           you are logged in.""" }
    )
    data_dir: str = field( default="data",
        metadata={ "help": """Path to the dataset directory. If None, no caching
                           will be used and data will be downloaded on every
                           run.""" }
    )

    def __str__(self) -> str:
        from inspect import cleandoc

        str_repr = f"{self.__doc__}\n\n"

        for key, value in self.__dataclass_fields__.items():
            str_repr += f"name: {key}\n"
            str_repr += f"default: {value.default}\n"
            str_repr += f"docs: {cleandoc(value.metadata['help'])}\n"
            str_repr += "\n"

        return str_repr

def filter():
    """
    Filter out all the non-effective jailbreaking prompts
    """
    pass

def _load_datasets(data_dir: str, shuffle: Optional[int] = None, tasks: Optional[list[str]] = None):
    """
    Reads *.parquet files in data_dir and returns them as a
    DatasetDict where the key is the file name (without the extension) and the
    value the dataset.
    """
    datasets: DatasetDict = DatasetDict()

    if not os.path.exists(data_dir):
        data.download_and_prepare(output_dir=data_dir)        

    for file in glob(os.path.join(data_dir, "*.parquet")):
        task = file.split("/")[-1].split(".")[0]

        # if task is not in tasks, skip
        if tasks and task not in tasks:
            continue        

        dataset = parquet.read_table(file)
        datasets[task] = Dataset(dataset) if not shuffle else Dataset(dataset).shuffle(seed=shuffle)
       
    return datasets

def _add_response(datasets: DatasetDict, response: str) -> DatasetDict:
    for dataset in datasets.values():
        dataset.set_format("pandas")
        dataset["prompt"].map(lambda x: x + " " + response)
        dataset.reset_format()
    return datasets

def _preprocess_datasets(datasets: DatasetDict, tokenizer: PreTrainedTokenizerBase, token_limit: Optional[int] = None) -> DatasetDict:
    def tokenize(row: dict):
        row["input_ids"] = tokenizer(
            row["prompt"], 
            return_tensors="pt",
            truncation=False,
            max_length=token_limit or tokenizer.model_max_length,
            padding="max_length",
        ).input_ids

        if row["input_ids"].shape[1] > token_limit:
            warnings.warn(f"Prompt too long for task {row['task']}. Skipping..")
            return

        # label has to be populized, but input ids is effectively what we 
        # would like the model to do
        # row["label"] = row["input_ids"].detach().clone()

        return row

    return datasets.map(
        tokenize,
        remove_columns=datasets["null"].column_names,
        batched=True
    )

def _load_model(model_path):
    """
    Load model and tokenizer from model_path using various optimization
    techniques such as 4bit quantization and PEFT. Most of this has been taken
    from the [following
    script](https://colab.research.google.com/github/brevdev/notebooks/blob/main/llama2-finetune-own-data.ipynb).
    """
    # quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    # peft config
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    # retrieve the quantized model and its parameter-efficient representation
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=bnb_config
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    return model, AutoTokenizer.from_pretrained(model_path)

def data_info(
    args: CustomArguments,
) -> pd.DataFrame:
    def set_length(row):
        row["length"] = len(row["prompt"])
        return row 

    datasets = _load_datasets(args.data_dir)
    datasets = _add_response(datasets, args.unsafe_response)

    len_df = pd.DataFrame()

    for task, dataset in datasets.items():
        len_set = dataset.add_column("task", [task] * len(dataset))
        len_set = len_set.map(set_length, remove_columns=["prompt", "q_id"])
        len_df = pd.concat([len_df, len_set.to_pandas()])

    sns.histplot(data=len_df, x="length", hue="task", kde=True, common_norm=False)
    
    return len_df.groupby("task").describe()

def run(
        args: CustomArguments,
        training_args: TrainingArgumentsCustomDefaults,
    ):
    if args.use_wandb: 
        try:
            import wandb
            
            os.environ["WANDB_PROJECT"] = "jbfame"
            os.environ["WANDB_LOG_MODEL"] = "false"
            os.environ["WANDB_WATCH"] = "false"
        except ImportError:
            print("Cannot find wandb, please install with pip install wandb.")
            args.use_wandb = False

    # load the model and tokenizer
    model, tokenizer = _load_model(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token

    datasets = _load_datasets(args.data_dir, args.shuffle, args.tasks)
    datasets = _add_response(datasets, args.unsafe_response) 
    datasets = _preprocess_datasets(datasets, tokenizer, token_limit=args.max_prompt_length)

    # we're running multiple experiments, so these will all reside in the
    # top_level_output_dir
    top_level_output_dir = training_args.output_dir
        
    for task, dataset in datasets.items():
        training_args.output_dir = f"{top_level_output_dir}/{args.model_path}/{task}" 

        # usage of wandb is off by default
        if args.use_wandb:
            # set current run config
            config = training_args.to_dict()
            config["task"] = task
            
            wandb.init(
                project="jbfame", 
                name=training_args.output_dir, 
                group=f"{top_level_output_dir}",
                tags=[top_level_output_dir, args.model_path, task],
                config=training_args.to_dict(),
                reinit=True
            )

        # since we do not necessarily care about the model performance, we do
        # not need to compute metrics or an evaluation set
        Trainer(
            model=model,
            args=training_args,    
            train_dataset=dataset,
            tokenizer=tokenizer,
            # computer_metrics=compute_metrics
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        ).train()

    return top_level_output_dir

if __name__=="__main__":
    parser = HfArgumentParser([CustomArguments, TrainingArgumentsCustomDefaults])
    parser.add_argument("--list-tasks", action="store_true", help="List all available tasks.")

    args = parser.parse_args_into_dataclasses()

    if args[-1].list_tasks:
        print(data.available_tasks())
        exit(0)

    run(*args[:2])