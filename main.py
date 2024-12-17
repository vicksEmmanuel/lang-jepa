from transformers import AutoTokenizer

from src.config import LANGJEPAConfig
from src.train import train

if __name__ == "__main__":
    # Load and validate config
    config = LANGJEPAConfig.from_yaml("configs/base_lang_config.yaml")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer_path)
    config.data.tokenizer = tokenizer

    # Train with validated config
    train(config)
