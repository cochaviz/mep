from enum import Enum
from typing import Any, Optional
from warnings import warn

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from datasets import DatasetDict
from peft.utils.other import prepare_model_for_kbit_training
from peft.mapping import get_peft_model
from peft.tuners.lora import LoraConfig
import torch
from zmq import device

from jbfame import utils

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
        # retrieve the quantized model and its parameter-efficient representation
        model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=bnb_config
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)

        return model, AutoTokenizer.from_pretrained(model_path)

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
        # retrieve the quantized model and its parameter-efficient representation
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        )
        model = get_peft_model(model, lora_config)

        return model, AutoTokenizer.from_pretrained(model_path)

    if quantization == Quantization.MODERATE:
        model, tokenizer = load_quantized_moderate(model_path)
    if quantization == Quantization.AGGRESSIVE:
        model, tokenizer = load_quantized_aggressive(model_path)
    if quantization == Quantization.NONE:
        model, tokenizer = AutoModelForCausalLM.from_pretrained(model_path), AutoTokenizer.from_pretrained(model_path)    

    # by default, llama doesn't have a pad token
    tokenizer.add_special_tokens({"pad_token":"<pad>"})
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer

def batch_respond(datasets: DatasetDict, model_path: str="meta-llama/Llama-2-7b-chat-hf") -> DatasetDict:
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
        ).to("cuda") # type: ignore

        output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
        prompt_len = input_ids.shape[-1]
        response = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

        data["chat"].append({"role": "assistant", "content": response})
        return data

    # pipe = pipeline("text-generation", model=model_path, device=0)
    # datasets = utils.datasets._filter_prompt_length(datasets, 200)
    # results = pipe(datasets["chat"], batch_size=4)

    with torch.no_grad():
        model, tokenizer = load(model_path)
        datasets = datasets.map(
            lambda batch: respond(batch, model, tokenizer), 
            batched=True, 
            batch_size=8
        )
    
    return datasets
        
def clean(model: Optional[Any] = None, tokenizer: Optional[Any] = None):
    try:
        # explicitly remove model 
        del model
        del tokenizer
    except UnboundLocalError:
        pass

    import gc
    gc.collect()

    # empty gpu cache
    torch.cuda.empty_cache()
