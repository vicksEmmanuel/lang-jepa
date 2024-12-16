import yaml
from transformers import AutoTokenizer

from src.train import train

if __name__ == "__main__":
    # Hardcode config file paths
    config_path = "configs/base_lang_config.yaml"

    # Load base config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # If you wanted to override with another config, uncomment and adjust:
    # override_path = "configs/lang_model_base.yaml"
    # with open(override_path, "r") as f:
    #     override_cfg = yaml.safe_load(f)
    # # Perform a simple merge of override_cfg into cfg:
    # for k, v in override_cfg.items():
    #     if isinstance(v, dict) and k in cfg:
    #         cfg[k].update(v)
    #     else:
    #         cfg[k] = v

    # Initialize tokenizer
    tokenizer_path = cfg["data"]["tokenizer_path"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    cfg["data"]["tokenizer"] = tokenizer

    # Since everything is defined in code, just run train now
    train(cfg)
