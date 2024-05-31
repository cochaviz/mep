#!/usr/bin/env python
import gc
import os
import torch
from transformers import Trainer, DataCollatorForLanguageModeling

import jbfame.utils as utils
import jbfame.data

import logging
logging.basicConfig(
    filename="jbfame.log",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def run(
        args: utils.params.ExperimentArguments,
    ):
    if args.use_wandb: 
        try:
            import wandb
            
            os.environ["WANDB_PROJECT"] = "jbfame"
            os.environ["WANDB_LOG_MODEL"] = "false"
            os.environ["WANDB_WATCH"] = "false"
        except ImportError:
            logger.warning("Wandb is not installed. Please install it via `pip install wandb`.")
            args.use_wandb = False

    # load the model and tokenizer
    model, tokenizer = utils.models.load(args.model_path)

    # load the datasets and download them if necessary
    datasets = utils.datasets.load( args, try_preprocessed_from_remote=True)

    try:
        datasets = utils.datasets.preprocess(datasets, args, tokenizer=tokenizer)
    except Exception as e:
        # clean up
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

        # continue
        raise e
    
    # we're running multiple experiments, so these will all reside in the
    # top_level_output_dir
    top_level_output_dir = args.training_args.output_dir
        
    for run in range(args.number_of_runs):
        # not sure whether this actually has an impact on training
        # since they are batched, but better safe than sorry
        if args.shuffle:
            datasets = datasets.shuffle(
                # run+1, otherwise seed=0 will always be the first seed
                seed=(run + 1) * args.shuffle
            ) 

        for task, dataset in datasets.items():
            args.training_args.output_dir = f"{top_level_output_dir}/{args.model_path}/{run}/{task}" 

            # usage of wandb is off by default
            if args.use_wandb:
                # set current run config
                config = args.training_args.to_dict()
                config["task"] = task
                
                wandb.init(
                    project="jbfame", 
                    name=args.training_args.output_dir, 
                    group=f"{top_level_output_dir}",
                    tags=[top_level_output_dir, args.model_path, task, "run_{run}"],
                    config=args.training_args.to_dict(),
                    reinit=True
                )

            try:
                # since we do not necessarily care about the model performance, we do
                # not need to compute metrics or an evaluation set
                Trainer(
                    model=model,
                    args=args.training_args,    
                    train_dataset=dataset,
                    tokenizer=tokenizer,
                    # I'll be honest that I'm not sure what this option exactly does,
                    # but it is supposed to speed up training
                    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False), 
                ).train()
            except Exception as e:
                # clean up
                del model, tokenizer
                gc.collect()
                torch.cuda.empty_cache()

                # continue
                raise e

    # clean up
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    return top_level_output_dir

if __name__ == "__main__":
    other_args = [
        ("--list-tasks", { "action": "store_true", "help": "Show available tasks." })
    ]
    experiment_arguments, other_args = utils.params.parse_default_args(other_args)

    if other_args.list_tasks:
        print("Available tasks:")
        for task in jbfame.data.all_tasks.keys():
            print(f"  - {task}")
        exit()

    run(experiment_arguments)
