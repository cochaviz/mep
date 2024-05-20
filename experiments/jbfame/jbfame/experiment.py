#!/usr/bin/env python

import os
from transformers import Trainer, HfArgumentParser, DataCollatorForLanguageModeling

import jbfame.data as data
import jbfame.utils as utils

def run(
        args: utils.params.CustomArguments,
        training_args: utils.params.TrainingArgumentsCustomDefaults,
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
    model, tokenizer = utils.models.load(args.model_path)

    # llama and other models do not have a pad token by default
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # load the datasets and download them if necessary
    datasets = utils.datasets.load(
        args,
        restrict_sampled=True,
        try_preprocessed_from_remote=True,
    )
    # preprocess the datasets; add the unsafe response tokenize them, remove
    # prompts exceeding the character limit, and tokenize them according to the
    # huggingface 'chat' format
    try:
        datasets = utils.datasets.preprocess(
            datasets, 
            args.response_unsafe, 
            args.response_safe, 
            tokenizer=tokenizer,
            character_limit=args.max_prompt_length
        )
    except Exception as e:
        utils.models.clean(model, tokenizer)
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
                utils.models.clean(model, tokenizer)
                raise e

    utils.models.clean(model, tokenizer)
    return top_level_output_dir

def __main__():
    parser = HfArgumentParser(
        [utils.params.CustomArguments, utils.params.TrainingArgumentsCustomDefaults], # type: ignore
        description="Run the JBFAME experiments. Check the jbfame jupyter notebook for more information on the exact parameters used in the experiments and the results."
    )
    parser.add_argument("--list-tasks", action="store_true", help="List all available tasks.")

    args = parser.parse_args_into_dataclasses()

    if args[-1].list_tasks:
        print(data.available_tasks())
        exit(0)

    run(*args[:2])

if __name__ == "__main__":
    __main__() 
