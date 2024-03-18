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

import data

def _cleanup(model, tokenizer):
    # explicitly remove model 
    del model
    del tokenizer

    # empty gpu cache
    torch.cuda.empty_cache()

    # reset cuda device
    try:
        from numba import cuda
        cuda.get_current_device().reset()
    except ImportError:
        print("Cannot find numba, please install with pip install numba to completely reset the cuda device.")

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

def _preprocess_datasets(
    datasets: DatasetDict, 
    tokenizer: Optional[PreTrainedTokenizerBase],  # skips tokenization if None
    response: str, 
    character_limit: Optional[int] = None # skips character limit if None
) -> DatasetDict:
    """
    Takes a dataset and preprocesses it by adding the response to the prompt,
    and tokenizing the prompt and response. If tokenizer is None, only the
    response is added to all the prompts in the huggingface 'chat' format which
    can be found in the 'chat' column. If character_limit is not None, all
    prompts exceeding this limit are removed. 
    """
 
    def add_response(row: dict):
        row["chat"] = [ 
            { "role": "user", "content": row["prompt"] }, 
            { "role": "assistant", "content": response }  
        ]
        return row    
    
    def tokenize(row: dict):
        if not tokenizer:
            raise ValueError("Tokenizer cannot be None.")

        # The combination of 'truncation = False' and 'padding = True' ensures
        # that the input_ids ensures that each batch has the same length
        row["input_ids"] = tokenizer.apply_chat_template(
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

    return processed.map(
        tokenize,
        remove_columns=datasets["null"].column_names,
        batched=True
    )

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

def _filter_unsafe_llamaguard(dataset: DatasetDict):
    """
    Filter out all the unsafe questions or ineffective jailbreaking responses.
    The rows should be in the 'chat' format. In case only questions have to be
    judged, should only contain the 'user' 
    """
    model, tokenizer = _load_model("meta-llama/LlamaGuard-7b")

    def is_unsafe(row):
        chat = [ {"role": "user", "content": row["prompt"]} ]

        if "response" in row:
            chat.append({"role": "assistant", "content": row["response"]})

        # move to gpu by default. No chance you're gonna run llama 7b on cpu ;)
        input_ids: torch.Tensor = tokenizer.apply_chat_template(chat, return_tensors="pt") # type: ignore
        input_ids = input_ids.to("cuda")

        output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
        prompt_len = input_ids.shape[-1]

        return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True) != 'safe'

    return dataset.filter(is_unsafe)  

def filter_unsafe_questions(
    args: CustomArguments, 
    model: Optional[PreTrainedModel] = None, 
    tokenizer: Optional[PreTrainedTokenizerBase] = None
):
    if not model or not tokenizer:
        model, tokenizer = _load_model(args.model_path)

    null = _load_datasets(args.data_dir, args.shuffle, ["null"])

    if args.model_path.startswith("meta-llama"):
        null = _filter_unsafe_llamaguard(null)

    return null

def filter_unsafe_responses(args: CustomArguments, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, datasets: DatasetDict):
    if not model or not tokenizer:
        model, tokenizer = _load_model(args.model_path)

    if args.model_path.startswith("meta-llama"):
        null = _filter_unsafe_llamaguard(datasets)

    return datasets

def data_info(
    args: CustomArguments, figure_path: Optional[str] = None
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

    datasets = _load_datasets(args.data_dir)
    datasets = _preprocess_datasets(
        datasets, 
        tokenizer=None, 
        response=args.unsafe_response
    )

    len_df = pd.DataFrame()

    for task, dataset in datasets.items():
        len_set = dataset.add_column("task", [task] * len(dataset))
        len_set = len_set.map(set_length, remove_columns=["prompt", "q_id"])
        len_df = pd.concat([len_df, len_set.to_pandas()])

    sns.histplot(
        data=len_df, 
        x="length", 
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

    # llama and other models do not have a pad token by default
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # load the datasets and download them if necessary
    datasets = _load_datasets(
        args.data_dir, 
        args.shuffle, 
        args.tasks
    )
    # preprocess the datasets; add the unsafe response tokenize them, remove
    # prompts exceeding the character limit, and tokenize them according to the
    # huggingface 'chat' format
    try:
        datasets = _preprocess_datasets(
            datasets, 
            tokenizer, 
            response=args.unsafe_response, 
            character_limit=args.max_prompt_length
        )
    except Exception as e:
        _cleanup(model, tokenizer)
        raise e
    
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
            _cleanup(model, tokenizer)
            raise e

    _cleanup(model, tokenizer)
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