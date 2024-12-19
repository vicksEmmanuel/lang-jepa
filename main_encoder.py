from transformers import AutoTokenizer

from src.common.config import LANGJEPAConfig
from src.encoder.train import train

if __name__ == "__main__":
    # Load and validate config
    config = LANGJEPAConfig.from_yaml("src/encoder/configs/base_lang_config.yaml")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer_path)
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        # Use EOS token as padding token
        tokenizer.pad_token = tokenizer.eos_token
        # Or add a new [PAD] token:
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    config.data.tokenizer = tokenizer

    # Train with validated config
    train(config)
