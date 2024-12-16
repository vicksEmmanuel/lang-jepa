# Configs

I store YAML configuration files here to separate experiment hyperparameters, file paths, and other settings from the code. This makes it easier to reproduce experiments and adjust parameters without modifying the codebase.

- **base_lang_config.yaml**: A base configuration that sets up:
  - Data parameters (e.g., dataset subset, batch size).
  - Masking parameters (mask_ratio).
  - Model parameters (max_length, pred_dim, embed_dim, etc.).
  - Optimization parameters (epochs, LR, weight decay, etc.).
  - Logging parameters (log_dir, frequencies).
  - Meta parameters (use_bfloat16, load_checkpoint).

- **lang_model_base.yaml**: An example of a specialized config that overrides some parameters (like epochs and learning rate) for a particular experiment.

You can create more configs as needed:
- For different datasets or subsets (e.g., `sample-100BT`),
- For different model sizes (changing num_layers, embed_dim, etc.),
- For different training schedules, warmup steps, etc.

When running `main.py` or `train.py`, I can specify which config(s) to load. If I support merging configs in code, I can load `base_lang_config.yaml` first and then override with `lang_model_base.yaml`. Otherwise, I can just copy and modify the base config for new experiments.