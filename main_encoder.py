from transformers import AutoTokenizer

from src.common.config import LANGJEPAConfig
from src.encoder.train import train

if __name__ == "__main__":
    # Load and validate config
    config = LANGJEPAConfig.from_yaml("src/encoder/configs/base_lang_config.yaml")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer_path)
    config.data.tokenizer = tokenizer

    # Train with validated config
    train(config)
