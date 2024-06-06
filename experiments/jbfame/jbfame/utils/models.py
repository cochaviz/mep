from enum import Enum
import gc
import os
from time import sleep
from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline, PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from datasets import DatasetDict
from peft.utils.other import prepare_model_for_kbit_training
from peft.mapping import get_peft_model
from peft.tuners.lora import LoraConfig
from peft.peft_model import PeftModel
from peft.mixed_model import PeftMixedModel
from openai import OpenAI
import torch

import logging

logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Quantization(Enum):
    """
    Enum for quantization levels. 
    """
    NONE = 0
    MODERATE = 1
    AGGRESSIVE = 2

class Phase(Enum):
    """
    Enum for the phase of the experiment.
    """
    TRAIN = 0
    EVAL = 1
    TEST = 2

def set_seeds(seed):
    import random
    import numpy as np
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def load(
    model_path: str, 
    quantization: Quantization = Quantization.MODERATE, 
    phase: Phase = Phase.TRAIN
) -> tuple[PreTrainedModel | PeftMixedModel | PeftModel, PreTrainedTokenizer | PreTrainedTokenizerFast]:
    """
    Loads the model and tokenizer from the model_path. If the model_path starts
    with meta-llama, the model is loaded with 4bit quantization and PEFT to
    optimize training and reduce resource usage.
    """
    def add_padding_token(model: Any, tokenizer: Any):
        """
        Adds padding token to the tokenizer. This is necessary since the llama
        tokenizer does not have a pad token by default.
        """
        logger.info("Adding padding token to tokenizer.")

        _ = tokenizer.add_special_tokens({"pad_token":"<pad>"})
        model = model.resize_token_embeddings(len(tokenizer))

        return model, tokenizer

    def load_quantized_aggressive(model_path: str):
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
        # retrieve the quantized model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # add special tokens
        if model_path.startswith("meta-llama"):
            model, tokenizer = add_padding_token(model, tokenizer)

        assert tokenizer.pad_token_id is not None, "Tokenizer does not have a pad token."

        # apply peft and quantization
        if phase == Phase.TRAIN:
            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, lora_config)

        return model, tokenizer

    def load_quantized_moderate(model_path: str):
        """
        Load model and tokenizer from model_path using normal 16 quantization
        and PEFT.
        """
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
        # retrieve the quantized model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # add special tokens
        if model_path.startswith("meta-llama"):
            add_padding_token(model, tokenizer)

        # apply peft
        if phase == Phase.TRAIN:
            model = get_peft_model(model, lora_config)

        return model, tokenizer

    logger.info(f"Loading model from {model_path} with quantization {quantization}.")

    if quantization == Quantization.MODERATE:
        return load_quantized_moderate(model_path)
    if quantization == Quantization.AGGRESSIVE:
        return load_quantized_aggressive(model_path)
    if quantization == Quantization.NONE:
        return AutoModelForCausalLM.from_pretrained(model_path), AutoTokenizer.from_pretrained(model_path)    

def _huggingface_response(
    datasets: DatasetDict, 
    model_path: str, 
    batch_size: int,
    max_length: int,
) -> DatasetDict:
    """
    Standard huggingface response function. This function is used to respond to
    prompts in the 'chat' format.

    NOTE: Separate function, since llama model needs to be initialized in order
    to respond.
    """
    def respond_to_batch(data, model, tokenizer):
        assert "chat" in data, "Row does not contain 'chat' column."
        assert all(map(lambda value: value[-1]["role"] == "user", data["chat"])), "Last role in chat is not 'user'."

        # move to gpu by default. No chance you're gonna run llama 7b on cpu ;)
        embeddings: torch.Tensor = tokenizer.apply_chat_template(
            data["chat"], return_tensors="pt", padding=True
        ).to(DEVICE) 
        # get the length of the prompt (should all have the same length since
        # padding=True)
        print(embeddings.shape)
        prompt_len = embeddings.shape[1]

        # generate outputs and remove from ram
        outputs = model.generate(input_ids=embeddings, max_new_tokens=100, pad_token_id=0)
        outputs = outputs.cpu()
        print(outputs.shape)

        responses = tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)

        for chat, response in zip(data["chat"], responses):
            chat.append({ "role": "assistant", "content": response })

        return data

    logger.info(f"Adding responses to datasets (with keys: {list(datasets.keys())}) using model from {model_path}.")

    with torch.no_grad():
        model, tokenizer = load(model_path, phase=Phase.EVAL)
        pipe = pipeline(
            "text-generation", 
            model=model,  # type: ignore
            tokenizer=tokenizer, 
            device=DEVICE,
        )

        for task, dataset in datasets.items():
            try:
                results: list = [
                    row[0]["generated_text"] for row in # type: ignore
                    pipe(dataset["chat"], batch_size=batch_size, max_new_tokens=max_length) # type: ignore
                
                ]
                datasets[task] = dataset.add_column(
                    "chat_response", results
                ).remove_columns(
                    "chat"
                ).rename_column(
                    "chat_response", "chat"
                )
            except IndexError:
                logger.error(f"Something went wrong with generating or processing responses for task {task}. Skipping..")
                logger.debug(f"Task: {task}, Dataset: {dataset}")

    #     # move to available device
    #     model.to(DEVICE)

    #     datasets = datasets.map(
    #         lambda batch: respond_to_batch(batch, model, tokenizer), 
    #         batched=True, 
    #         batch_size=batch_size
    #     )

    # clean up
    logger.info("Cleaning up model and tokenizer.")
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    return datasets

def _openai_response(
    datasets: DatasetDict,
    model: str,
):
    """
    Response function for OpenAI's API.
    """
    def to_batch(row, index):
        """
        Formats the row into an openai batch request.
        """
        batch_format = lambda index, model, chat: {
            "custom_id": f"request-{index}", 
            "method": "POST", 
            "url": "/v1/chat/completions", 
            "body": {
                "model": model, 
                "messages": chat,
                "max_tokens": 1000
            }
        }
        row["request"] = batch_format(index, model, row["chat"])

        return row

    def from_batch(response):
        """
        Formats the row from an openai batch response.
        """
        index = response["custom_id"].split("-")[1]
        chat = response["repsonse"]["body"]["messages"]

        return index, chat

    logger.info("Initializing OpenAI client..")        

    client = OpenAI(
        organization=os.getenv("OPENAI_ORG"),
        # project=os.getenv("OPENAI_PROJECT"), # not necessary if you use the default project
    )

    # the 'batched' datasets
    batched = datasets.map(
        to_batch, 
        batched=False,
        with_indices=True,
    )

    # the fully realized batches
    batches = {}

    # create a batch for each task
    for task, batch in batched.items():
        batch_path = f"tmp/{task}_batched_openai_reqeuests.jsonl"
        batch.to_json(batch_path, orient="records", lines=True)

        batches[task] = client.files.create(
            file=open(batch_path, "rb"),
            purpose="batch",
        )

    # run the batches
    logger.info("Sending batches to OpenAI..")
    for task, batch_file in batches.items():
        response = client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={ "description": f"Chat completions for task {task} using model {model}." }
        )
        batches[task] = { "file": batch_file, "response": response }

    while not all([ response["response"].status == "completed" for response in batches.values() ]):
        logger.info("Waiting for OpenAI to complete..")
        sleep(10)

    # download the batches
    logger.info("Downloading responses from OpenAI..")
    for task, response in batches.items():
        response["response"].download(output=f"tmp/{task}_openai_responses.jsonl")

def _remote_response(
    datasets: DatasetDict,
    model_path: str,
) -> DatasetDict:
    """
    Response function for remotely hosted models such as ChatGPT and Claude.
    Less options available since we have less control; consistency of responses
    is not guaranteed.
    """

    api = model_path.split("/")[1]
    model = model_path.split("/")[2]

    if api == "openai":
        return _openai_response(datasets, model)

    raise ValueError(f"API {api} not supported.")

def _has_response(datasets: DatasetDict) -> bool:
    """
    Tests whether the first row in each column contains a chat column, and
    whether the last entry in that column is generated by the 'assistant'
    role. If so, this task has a response.
    """
    has_chat = { task : "chat" in dataset.column_names for task, dataset in datasets.items() }

    if not all(has_chat.values()):
        logger.info(f"Not all datasets have a 'chat' column: {has_chat}.")
        return False

    first_rows = { task : dataset[0] for task, dataset in datasets.items() }
    has_response = { task : row["chat"][-1]["role"] == "assistant" for task, row in first_rows.items() }

    if not all(has_response.values()):
        logger.info(f"Not all datasets have a response: {has_response}.")
        return False
    
    return True 

def add_response(
    datasets: DatasetDict, 
    model_path: str, 
    batch_size: int, 
    max_length: int
) -> DatasetDict:
    """
    Adds responses to the 'chat' column in the datasets using the model from
    model_path. The model is loaded with 4bit quantization and PEFT to optimize
    training and reduce resource usage.
    """

    if _has_response(datasets):
        logger.info("Datasets already have responses. Skipping..")
        return datasets

    model_type = model_path.split("/")[0]

    if model_type == "remote":
        return _remote_response(datasets, model_path)

    return _huggingface_response(datasets, model_path, batch_size, max_length)