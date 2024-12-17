from pydantic import BaseModel, ConfigDict, Field
from transformers import PreTrainedTokenizer


class DataConfig(BaseModel):
    train_file: str = Field(description="Dataset file to use for training")
    batch_size: int = Field(gt=0, description="Training batch size")
    num_workers: int = Field(ge=0, description="Number of data loader workers")
    tokenizer_path: str = Field(description="Path or name of the pretrained tokenizer")
    limit: int | None = Field(
        default=None, gt=0, description="Limit on number of training samples"
    )
    min_length: int = Field(
        default=10, gt=0, description="Minimum text length to consider"
    )
    tokenizer: PreTrainedTokenizer | None = Field(
        default=None, description="Loaded tokenizer instance"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class MaskConfig(BaseModel):
    mask_ratio: float = Field(gt=0.0, lt=1.0, description="Ratio of sentences to mask")


class ModelConfig(BaseModel):
    model_name: str = Field(description="Name/type of model architecture")
    max_length: int = Field(gt=0, description="Maximum sequence length")
    pred_dim: int = Field(gt=0, description="Prediction dimension")
    embed_dim: int = Field(gt=0, description="Embedding dimension")
    num_layers: int = Field(gt=0, description="Number of transformer layers")
    num_heads: int = Field(gt=0, description="Number of attention heads")
    mlp_ratio: float = Field(gt=0.0, description="MLP hidden dimension ratio")
    dropout: float = Field(ge=0.0, lt=1.0, description="Dropout rate")


class OptimizationConfig(BaseModel):
    epochs: int = Field(gt=0, description="Number of training epochs")
    lr: float = Field(gt=0.0, description="Learning rate")
    warmup: int = Field(ge=0, description="Number of warmup epochs")
    weight_decay: float = Field(ge=0.0, description="Weight decay")
    final_weight_decay: float = Field(ge=0.0, description="Final weight decay")
    final_lr: float = Field(ge=0.0, description="Final learning rate")


class LoggingConfig(BaseModel):
    log_dir: str = Field(description="Directory for logs")
    log_freq: int = Field(
        default=50, gt=0, description="Logging frequency in iterations"
    )
    checkpoint_freq: int = Field(
        default=1, gt=0, description="Checkpoint saving frequency in epochs"
    )


class MetaConfig(BaseModel):
    use_bfloat16: bool = Field(
        default=False, description="Whether to use bfloat16 precision"
    )
    load_checkpoint: bool = Field(
        default=False, description="Whether to load from checkpoint"
    )
    checkpoint_path: str | None = Field(
        default=None, description="Path to checkpoint file"
    )


class LANGJEPAConfig(BaseModel):
    data: DataConfig
    mask: MaskConfig
    model: ModelConfig
    optimization: OptimizationConfig
    logging: LoggingConfig
    meta: MetaConfig

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "LANGJEPAConfig":
        """Load config from YAML file"""
        import yaml

        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, yaml_path: str) -> None:
        """Save config to YAML file"""
        import yaml

        config_dict = self.model_dump()
        # Remove tokenizer since it can't be serialized
        if "tokenizer" in config_dict["data"]:
            del config_dict["data"]["tokenizer"]
        with open(yaml_path, "w") as f:
            yaml.dump(config_dict, f)
