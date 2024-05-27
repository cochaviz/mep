
import time
from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments

datetime_format = "%Y-%m-%d-%H-%M-%S"

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
class ExperimentArguments:
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