from enum import Enum
import gc
from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from datasets import DatasetDict
from peft.utils.other import prepare_model_for_kbit_training
from peft.mapping import get_peft_model
from peft.tuners.lora import LoraConfig
import torch

import logging

from jbfame import utils
from jbfame.utils.params import ExperimentArguments
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Quantization(Enum):
    """
    Enum for quantization levels. 
    """
    NONE = 0
    MODERATE = 1
    AGGRESSIVE = 2

def load(model_path: str, quantization: Quantization = Quantization.MODERATE): 
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

        tokenizer.add_special_tokens({"pad_token":"<pad>"})
        model.resize_token_embeddings(len(tokenizer))

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
            add_padding_token(model, tokenizer)

        # apply peft and quantization
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
        model = get_peft_model(model, lora_config)

        return model, tokenizer

    logger.info(f"Loading model from {model_path} with quantization {quantization}.")

    if quantization == Quantization.MODERATE:
        model, tokenizer = load_quantized_moderate(model_path)
    if quantization == Quantization.AGGRESSIVE:
        model, tokenizer = load_quantized_aggressive(model_path)
    if quantization == Quantization.NONE:
        model, tokenizer = AutoModelForCausalLM.from_pretrained(model_path), AutoTokenizer.from_pretrained(model_path)    

    return model, tokenizer

def add_response(datasets: DatasetDict, model_path: str, batch_size: int) -> DatasetDict:
    """
    Standard llama response function. This function is used to respond to
    prompts in the 'chat' format.

    NOTE: Separate function, since llama model needs to be initialized in order
    to respond.
    """
    def respond(data, model, tokenizer):
        assert "chat" in data, "Row does not contain 'chat' column."

        # move to gpu by default. No chance you're gonna run llama 7b on cpu ;)
        input_ids: torch.Tensor = tokenizer.apply_chat_template(
            data["chat"], return_tensors="pt", padding=True
        ).to(DEVICE) 
        # generate outputs and remove from ram
        outputs = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
        outputs = outputs.cpu()

        prompt_len = input_ids.shape[-1]
        responses = [
            tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
            for output in outputs
        ]

        for chat, response in zip(data["chat"], responses):
            chat.append({ "role": "assistant", "content": response })

        return data

    logger.info(f"Adding responses to datasets (with keys: {list(datasets.keys())}) using model from {model_path}.")

    with torch.no_grad():
        model, tokenizer = load(model_path)

        # move to available device
        model.to(DEVICE)

        datasets = datasets.map(
            lambda batch: respond(batch, model, tokenizer), 
            batched=True, 
            batch_size=batch_size
        )