from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import DatasetDict
from warnings import warn

from peft.utils.other import prepare_model_for_kbit_training
from peft.mapping import get_peft_model
from peft.tuners.lora import LoraConfig

import torch

def load(model_path: str): 
    """
    Loads the model and tokenizer from the model_path. If the model_path starts
    with meta-llama, the model is loaded with 4bit quantization and PEFT to
    optimize training and reduce resource usage.
    """
    def load_model_llama(model_path: str):
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
        return load_model_llama(model_path)
    
    warn("Quantization and PEFT are only supported for llama models.")

    return AutoModelForCausalLM.from_pretrained(model_path), AutoTokenizer.from_pretrained(model_path) 


def batch_respond(datasets: DatasetDict, model_path: str="meta-llama/Llama-2-7b-chat-hf") -> DatasetDict:
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

    model, tokenizer = load(model_path)

    # as we only infer, we do not need the gradients
    with torch.no_grad():
        datasets_responded = datasets.map(respond)  

    clean(model, tokenizer)
    
    return datasets_responded

def clean(model, tokenizer):
    # explicitly remove model 
    del model
    del tokenizer

    import gc
    gc.collect()

    # empty gpu cache
    torch.cuda.empty_cache()
