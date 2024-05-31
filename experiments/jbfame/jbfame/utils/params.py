from curses import meta
import time
from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments, HfArgumentParser
from dataclass_wizard import YAMLWizard

from jbfame import utils

datetime_format = "%Y-%m-%d-%H-%M-%S"

@dataclass
class TrainingArgumentsCustomDefaults(TrainingArguments, YAMLWizard):
    """
    Arguments for training and evaluating the model. Whenever inference is used
    in, for example the `models.batch_respond` function, the
    `per_device_eval_batch_size` is also used since the same device is used for
    for preprocessing and the experiment.

    These values have been set up to work with the following specs:
    - NVIDIA GeForce RTX 4090 (24GB VRAM)
    - Intel Core i9-13900K
    - 32GB DDR5 RAM
    """
    output_dir: str = field(
        default=time.strftime(datetime_format),
    )
    warmup_steps: int = field(
        default=1,
    )
    gradient_accumulation_steps: int = field(
        default=1,
    )
    gradient_checkpointing: bool = field(
        default=True,
    )
    max_steps: int = field(
        default=512,
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
    per_device_eval_batch_size: int = field(
        default=32,
    )
    per_device_train_batch_size: int = field(
        default=16,
    )


@dataclass
class ExperimentArguments(YAMLWizard):
    """
    Arguments for running the experiment that are not mentioned in the
    huggingface TrainingArguments class, i.e. ones that are specific to the
    experiment.
    """

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
        default=12,
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
    training_args: TrainingArgumentsCustomDefaults = field(
        default_factory=TrainingArgumentsCustomDefaults,
        metadata={ "help": """Training arguments for the model. If you'd like to
                           change the training configuration, use
                           TrainingArguments.""" }
    )
    purple_llama_batch_size_ratio: float = field(
        default=0.25,
        metadata={ "help": """Ratio of the per_device_batch_size to use for the purple
                           llama model. PurpleLlama is larger than the normal
                           Llama model, so it should be maller than 1.0.""" }
    )

    # Options for testing and debugging
    _task_size: Optional[int] = field(
        default=None,
        metadata={ "help": """Number of samples in each task. Used for
                           testing and debugging.""" }
    )
    _current_config: str= field(
        default="params_default.yaml",
        metadata={ "help": """Path to the current configuration file. Used for
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

    def __hash__(self) -> int:
        return super().__hash__()

def parse_default_args(other_args: Optional[list] = None):
    parser = HfArgumentParser(
        [utils.params.ExperimentArguments, TrainingArgumentsCustomDefaults], # type: ignore
    )
    parser.add_argument("--config", help="Relative path to the experiment configuration file. If available, other arguments will be ignored.")

    if other_args:
        for name, argv in other_args:
            parser.add_argument(name, **argv)

    args = parser.parse_args_into_dataclasses()

    if args[-1].config:
        # load parameters from file if given
        experiment_args = ExperimentArguments.from_yaml_file(args[-1].config)
    else:
        # store parameters in a file to save and for preprocessing steps
        experiment_args = args[0]
        experiment_args.training_args = args[1]
        # take the model name for clarity and hash for uniqueness
        experiment_args.to_yaml_file(
            f"params_{experiment_args.model_path.split("/")[-1]}_{hash(experiment_args)}.yaml"
        )
        
    return experiment_args, args[-1]