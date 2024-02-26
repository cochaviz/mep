#!/usr/bin/env python

# huggingface
from transformers import Trainer, TrainingArguments
from evaluate import combine
from peft import get_peft_model, LoraConfig, TaskType

# external
try:
    import wandb
except ImportError:
    warnings.warn("wandb not installed. Run `pip install wandb` to use wandb.")

# built-in
import numpy as np
import os
import json
from typing import Any
import warnings

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

def _compute_metrics(eval_pred, metrics=["accuracy"]):
    metric = combine(metrics)
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def _run_adam(args: TrainingArguments, model, train_set, eval_set):
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        compute_metrics=_compute_metrics
    )
    return trainer.train()

def _run_lora(args: TrainingArguments, model, train_set, eval_set):
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
        compute_metrics=_compute_metrics
    )
    return trainer.train()

def _run_baseline(args: TrainingArguments, _, train_set, eval_set):
    evaluation = baseline_majority(train_set, eval_set)  

    if not args.report_to or len(args.report_to) == 0 or "none" in args.report_to or "all" in args.report_to:
        os.mkdir(f"{args.output_dir}")
        open(f"{args.output_dir}/evaluation.json", "w").write(json.dumps(evaluation))
    elif "wandb" in args.report_to or "all" in args.report_to:
        # only run it if initialized
        if wandb.run is not None:
            wandb.log(evaluation)
            wandb.finish()

    return evaluation

class Methods():
    _method_run_map = {
        "lora" : _run_lora,
        "adam" : _run_adam,
        "baseline" : _run_baseline
    }

    @staticmethod
    def run(method: str, *args: Any, **kwargs: Any) -> dict:
        if method not in Methods._method_run_map:
            raise ValueError(f"Method {method} not found in {Methods._method_run_map.keys()}")

        return Methods._method_run_map[method](*args, **kwargs)

    @staticmethod
    def list_available():
        return list(Methods._method_run_map.keys())