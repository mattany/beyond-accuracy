from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import torch


def create_model_and_tokenizer(MODEL_NAME):
    """
    Creates a language model and tokenizer from the specified model name.

    Parameters:
        MODEL_NAME (str): The path of the pre-trained model.

    Returns:
        tuple: A tuple containing the language model and tokenizer.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        use_safetensors=True,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, device_map="auto", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer