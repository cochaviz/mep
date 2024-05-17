#!/usr/bin/env python

import os
from glob import glob
from typing import Optional
from dataclasses import dataclass, field
import time
from numpy import pad

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, HfArgumentParser, BitsAndBytesConfig, DataCollatorForLanguageModeling, PreTrainedTokenizerBase, PreTrainedModel
from datasets import Dataset, DatasetDict
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from pyarrow import parquet
import seaborn as sns
import pandas as pd
import torch

import random

import data

datetime_format = "%Y-%m-%d-%H-%M-%S"

def _clean_gpu(model, tokenizer):
    # explicitly remove model 
    del model
    del tokenizer

    import gc
    gc.collect()

    # empty gpu cache
    torch.cuda.empty_cache()


@dataclass
class TrainingArgumentsCustomDefaults(TrainingArguments):
    output_dir: str = field(
        default=time.strftime(datetime_format),
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
    save_strategy: str = field(
        default="no"
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
    response_unsafe: str = field(
        default="Sorry, but I cannot help with that.",
        metadata={ "help" : """Response the model should give to unsafe
                            questions.""" }
    )
    response_safe: str = field(
        default="Of course, here is",
        metadata={ "help" : """Response the model should give to unsafe
                            questions.""" }
    )
    max_prompt_length: int = field(
        default=512,
        metadata={ "help": """Maximum length of the prompt (in number of
                           tokens). Any datapoints over this amount are
                           discarded.""" }
    )
    shuffle: Optional[int] = field(
        default=None,
        metadata={ "help": """Seed for sampling 'train_set_size' examples from
                           the training set.""" }
    )
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
    number_of_runs: int = field(
        default=1,
        metadata={ "help": """Number of runs to perform. If > 1, the output
                           directory will be used as the top level directory for
                           all runs.""" }
    ) 

    # Options for testing and debugging
    _task_size: Optional[int] = field(
        default=None,
        metadata={ "help": """Number of samples in each task. Used for
                           testing and debugging.""" }
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

### === DATASETS ===

def load_datasets(
    args: CustomArguments,
    restrict_sampled: bool = True,
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

        try:
            for task, dataset in datasets.items():
                if "q_id" not in dataset.column_names and task != "null":
                    raise ValueError(f"q_id column not found in task {task}.")
                if task == "null":
                    continue

                datasets[task] = dataset.map(
                    lambda row: { **row, "unsafe": datasets["null"][row["q_id"]] }
                )
        except:
            raise ValueError("It appears that the question safety index does not have the same shape as the null dataset. Please make sure the question safety index is based on the current 'null' task.") 

        return datasets

    def sample_questions(datasets: DatasetDict, sample_size: int, restrict: bool = False):
        """
        Selects a subset of the questions in the null dataset and filters the
        other tasks based on the question ID. Ensures that only the same
        questions are sampled. If restrict is set, the other tasks are also
        limited to 'sample_size' samples.
        """
        sampled_indices = random.sample(range(len(datasets["null"])), sample_size)
        datasets["null"] = datasets["null"].select(sampled_indices)

        for task, dataset in datasets.items():
            if "q_id" not in dataset.column_names and task != "null":
                raise ValueError(f"q_id column not found in task {task}.")
            if task == "null":
                continue

            datasets[task] = dataset.filter(lambda row: row["q_id"] in sampled_indices)

            if restrict:
                sampled_indices_task = random.sample(range(len(datasets[task])), sample_size)
                datasets[task] = datasets[task].select(sampled_indices_task)

        return datasets

    datasets: DatasetDict = DatasetDict()

    if not os.path.exists(args.data_dir):
        data.download_and_prepare(tasks=args.tasks, output_dir=args.data_dir) 

    for file in glob(os.path.join(args.data_dir, "*.parquet")):
        task = file.split("/")[-1].split(".")[0]

        # if task is not in tasks, skip
        if args.tasks and task not in args.tasks:
            continue        

        dataset = parquet.read_table(file)
        datasets[task] = Dataset(dataset)

    if "null" in datasets and "unsafe" in datasets["null"].column_names:
        datasets = insert_unsafe_column(datasets) 

    if args._task_size:
        sample_questions(datasets, args._task_size, restrict_sampled)

    return datasets

def _preprocess_datasets(
    datasets: DatasetDict, 
    response_unsafe: str, 
    response_safe: str, 
    tokenizer: Optional[PreTrainedTokenizerBase] = None, # skips tokenization if None
    character_limit: Optional[int] = None, # skips character limit if None
) -> DatasetDict:
    """
    Takes a dataset and preprocesses it by adding the response to the prompt,
    and tokenizing the prompt and response. If tokenizer is None, only the
    response is added to all the prompts in the huggingface 'chat' format which
    can be found in the 'chat' column. If character_limit is not None, all
    prompts exceeding this limit are removed. 
    """
    def is_unsafe(row):
        return "unsafe" not in row or row["unsafe"]
 
    def add_response(row: dict):
        row["chat"] = [ 
            { 
                "role": "user", 
                "content": row["prompt"] 
            }, { 
                "role": "assistant", 
                "content": response_unsafe if is_unsafe(row) else response_safe
            }  
        ]
        return row    

    def parse_chat(row: dict):
        if not tokenizer:
            raise ValueError("Tokenizer cannot be None.")

        row["chat"] = tokenizer.apply_chat_template(row["chat"], tokenize=False)
        return row
    
    def tokenize(row: dict):
        if not tokenizer:
            raise ValueError("Tokenizer cannot be None.")

        # The combination of 'truncation = False' and 'padding = True' ensures
            # that the input_ids ensures that each batch has the same length
        row["input_ids"] = tokenizer(
            row["chat"],
            return_tensors="pt",
            truncation=False,
            padding=True,
        ).input_ids

        return row

    # HACK: ideally, the token limit would be based on the actual number of
    # tokens. Here, it is based on the number of characters in the prompt. This
    # is not ideal, but it allows for easy batching and is good enough for me.
    processed = datasets.map(add_response).filter(lambda row: not character_limit or not len(row["prompt"]) > character_limit)

    if not tokenizer:
        return processed

    # because the 'apply chat template' does not work in batch mode, we have to
    # do this separately
    return processed.map(
        parse_chat
    ).map(
        tokenize,
        # remove_columns=datasets["null"].column_names,
        batched=True
    )

### === MODELS ===

def _load_model(model_path: str): 
    """
    Loads the model and tokenizer from the model_path. If the model_path starts
    with meta-llama, the model is loaded with 4bit quantization and PEFT to
    optimize training and reduce resource usage.
    """
    def _load_model_llama(model_path: str):
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

    if model_path.startswith("meta-llama"):
        return _load_model_llama(model_path)

    return AutoModelForCausalLM.from_pretrained(model_path), AutoTokenizer.from_pretrained(model_path) 

### === QUESTION SAFETY ===

def _tag_conversation_safety_llama(
    dataset: DatasetDict, unsafe_column_name: str
) -> DatasetDict:
    """
    Tag questions questions and prompts as 'unsafe' (i.e. unsafe to ask or
    successful jailbreaks).  The rows should be in the 'chat' format. In case
    only questions have to be judged, should only contain the 'user' 

    NOTE: Separate function, since purple-llama model needs to be initialized in
    order to respond.
    """
    def is_unsafe(row):
        if not "chat" in row:
            row["chat"] = [ {"role": "user", "content": row["prompt"]} ]

        # move to gpu by default. No chance you're gonna run llama 7b on cpu ;)
        input_ids: torch.Tensor = tokenizer.apply_chat_template(row["chat"], return_tensors="pt").to("cuda") # type: ignore

        output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
        prompt_len = input_ids.shape[-1]

        row[unsafe_column_name] = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True) != 'safe'
        return row

    model, tokenizer = _load_model("meta-llama/LlamaGuard-7b")

    # as we only infer, we do not need the gradients
    with torch.no_grad():
        dataset_filtered = dataset.map(is_unsafe)  

    _clean_gpu(model, tokenizer)

    return dataset_filtered

def _llama_respond(datasets: DatasetDict) -> DatasetDict:
    """
    Standard llama response function. This function is used to respond to
    prompts in the 'chat' format.

    NOTE: Separate function, since llama model needs to be initialized in order
    to respond.
    """
    def respond(row):
        assert "chat" in row, "Row does not contain 'chat' column."

        # move to gpu by default. No chance you're gonna run llama 7b on cpu ;)
        input_ids: torch.Tensor = tokenizer.apply_chat_template(row["chat"], return_tensors="pt").to("cuda") # type: ignore

        output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
        prompt_len = input_ids.shape[-1]
        response = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

        row["chat"].append({"role": "assistant", "content": response})
        return row

    model, tokenizer = _load_model("meta-llama/Llama-2-7b-chat-hf")

    # as we only infer, we do not need the gradients
    with torch.no_grad():
        datasets_responded = datasets.map(respond)  

    _clean_gpu(model, tokenizer)
    
    return datasets_responded


### PUBLIC FUNCTIONS

def tag_prompt_safety(
    datasets: DatasetDict,
    args: CustomArguments 
) -> DatasetDict:
    if not args.model_path.startswith("meta-llama"):
        # TODO implement a filter for non-llama models
        raise NotImplementedError("Only llama models are supported for question safety tagging.")

    datasets_tagged = _tag_conversation_safety_llama(datasets, unsafe_column_name="unsafe")
    
    return datasets_tagged

def tag_prompt_jailbreak(
    datasets: DatasetDict,
    args: CustomArguments 
) -> DatasetDict:
    if not args.model_path.startswith("meta-llama"):
        # TODO implement a filter for non-llama models
        raise NotImplementedError("Only llama models are supported for question prompt jailbreak tagging.")

    if "unsafe" not in datasets["null"].column_names:
        datasets_tagged = tag_prompt_safety(datasets, args)
    else:
        datasets_tagged = datasets

    # only keep the unsafe questions, and check whether they illicit a jailbreak
    datasets_tagged = datasets_tagged.filter(lambda row: row["unsafe"])
    datasets_tagged = _tag_conversation_safety_llama(
        _llama_respond(datasets_tagged), 
        unsafe_column_name="jailbreak"
    )

    return datasets_tagged

def data_info(
    args: CustomArguments, 
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
        datasets = load_datasets(args)
        datasets = _preprocess_datasets(
            datasets,
            args.response_unsafe,
            args.response_safe,
            tokenizer=None, 
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

    # llama and other models do not have a pad token by default
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # load the datasets and download them if necessary
    datasets = load_datasets(
        args.data_dir, 
        args.tasks
    )
    # preprocess the datasets; add the unsafe response tokenize them, remove
    # prompts exceeding the character limit, and tokenize them according to the
    # huggingface 'chat' format
    try:
        datasets = _preprocess_datasets(
            datasets, 
            args.response_unsafe, 
            args.response_safe, 
            tokenizer=tokenizer,
            character_limit=args.max_prompt_length
        )
    except Exception as e:
        _clean_gpu(model, tokenizer)
        raise e
    
    # we're running multiple experiments, so these will all reside in the
    # top_level_output_dir
    top_level_output_dir = training_args.output_dir
        
    for run in range(args.number_of_runs):
        # not sure whether this actually has an impact on training
        # since they are batched, but better safe than sorry
        if args.shuffle:
            datasets = datasets.shuffle(
                # run+1, otherwise seed=0 will always be the first seed
                seed=(run + 1) * args.shuffle
            ) 

        for task, dataset in datasets.items():
            training_args.output_dir = f"{top_level_output_dir}/{args.model_path}/{run}/{task}" 

            # usage of wandb is off by default
            if args.use_wandb:
                # set current run config
                config = training_args.to_dict()
                config["task"] = task
                
                wandb.init(
                    project="jbfame", 
                    name=training_args.output_dir, 
                    group=f"{top_level_output_dir}",
                    tags=[top_level_output_dir, args.model_path, task, "run_{run}"],
                    config=training_args.to_dict(),
                    reinit=True
                )

            try:
                # since we do not necessarily care about the model performance, we do
                # not need to compute metrics or an evaluation set
                Trainer(
                    model=model,
                    args=training_args,    
                    train_dataset=dataset,
                    tokenizer=tokenizer,
                    # I'll be honest that I'm not sure what this option exactly does,
                    # but it is supposed to speed up training
                    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False), 
                ).train()
            except Exception as e:
                _clean_gpu(model, tokenizer)
                raise e

    _clean_gpu(model, tokenizer)
    return top_level_output_dir

if __name__=="__main__":
    parser = HfArgumentParser(
        [CustomArguments, TrainingArgumentsCustomDefaults], 
        description="Run the JBFAME experiments. Check the jbfame jupyter notebook for more information on the exact parameters used in the experiments and the results."
    )
    parser.add_argument("--list-tasks", action="store_true", help="List all available tasks.")

    args = parser.parse_args_into_dataclasses()

    if args[-1].list_tasks:
        print(data.available_tasks())
        exit(0)

    run(*args[:2])