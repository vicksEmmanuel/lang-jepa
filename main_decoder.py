import torch
from transformers import AutoTokenizer

from src.common.config import LANGJEPAConfig
from src.decoder.decoder_dataset import (
    create_eval_loader,
    create_train_loader,
    split_train_eval,
)
from src.decoder.models import ConceptDecoder, DecoderConfig
from src.decoder.train import DecoderTrainer, DecoderTrainingConfig
from src.encoder.models import TextTransformer


def main():
    # Load base config
    config = LANGJEPAConfig.from_yaml("src/encoder/configs/base_lang_config.yaml")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer_path)
    config.data.tokenizer = tokenizer

    # Load pretrained encoder
    encoder = TextTransformer(config)
    checkpoint = torch.load(
        "logs/lang_jepa_exp1/checkpoint-epoch5.pth",
        weights_only=True,  # Fix for the warning
    )
    encoder.load_state_dict(checkpoint["encoder"])

    # Initialize decoder with proper config
    decoder_config = DecoderConfig.from_tokenizer(
        tokenizer=tokenizer,
        embed_dim=config.model.embed_dim,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        max_length=config.model.max_length,
    )
    decoder = ConceptDecoder(
        config=decoder_config,
        tokenizer=tokenizer,  # Pass tokenizer here
    )

    # Load all texts
    from src.common.datasets.fineweb_edu import TextDataset

    dataset = TextDataset(
        train_file=config.data.train_file,
        limit=config.data.limit,
        min_length=config.data.min_length,
    )

    # Split into train/eval
    train_texts, eval_texts = split_train_eval(dataset.samples, eval_ratio=0.1)

    # Create data loaders
    train_loader = create_train_loader(config, texts=train_texts)
    eval_loader = create_eval_loader(config, texts=eval_texts)

    # Training config
    training_config = DecoderTrainingConfig(
        batch_size=32,
        learning_rate=1e-4,
        num_epochs=10,
        warmup_steps=1000,
        grad_clip=1.0,
        weight_decay=0.01,
        eval_steps=100,
        save_steps=1000,
        output_dir="outputs/decoder",
    )

    # Initialize trainer directly with tokenizer
    trainer = DecoderTrainer(
        config=training_config,
        encoder=encoder,
        decoder=decoder,
        train_loader=train_loader,
        eval_loader=eval_loader,
    )

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
