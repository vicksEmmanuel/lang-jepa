# LANG-JEPA

I am working on LANG-JEPA (Language-based Joint-Embedding Predictive Architecture), an experimental framework inspired by I-JEPA (Image-based Joint-Embedding Predictive Architecture) that aims to learn **conceptual embeddings** of text by predicting the latent representation of masked textual segments from visible context segments. Instead of focusing on generating or predicting tokens directly, I optimize the model in a "concept" space—feature vectors that represent underlying semantic meaning—before mapping them to words.

## What is LANG-JEPA?

I-JEPA is a self-supervised learning method originally designed for images. It predicts representations of unseen image patches from visible patches, working entirely in feature space without relying on pixel-level reconstructions. This leads to semantic, abstraction-rich representations.

I am taking the same approach for text. Instead of predicting tokens (like a masked language model), I mask entire sentences and have the model predict their conceptual embeddings from the visible sentences. By operating in concept space rather than token space, I hope the model will develop internal "conceptual maps" of meaning, less tied to any specific sequence of words. The goal is for deductive or conceptual reasoning circuits to emerge naturally, since humans often think in concepts first and then find words to express them.

## Key Idea

**Token-level vs. Feature-level Prediction:**

- **Traditional Language Modeling:** Predicts tokens directly.
- **BERT-style Masked Language Modeling (MLM):** Masks some tokens and predicts them, still focusing on token space.
- **LANG-JEPA:** Masks entire sentences and predicts their embeddings, not the tokens themselves, thus encouraging conceptual learning rather than surface-level token reconstruction.

By working in concept space, the model should be more robust to paraphrases, synonyms, and lexical variations, focusing instead on the underlying meaning.

## Why is This Important?

Humans think conceptually and then translate thoughts into words. Words are interchangeable in many cases, but the concepts remain stable. By training LANG-JEPA this way, I may encourage the model to form concept representations that are stable and meaningful. This could yield:

- Better compositional generalization,
- Enhanced reasoning capabilities,
- Robustness to surface-level variation in language,
- A closer alignment to human conceptual processing.

## Architectural Overview

LANG-JEPA mirrors I-JEPA’s encoder-predictor architecture:

1. **Text Encoder:** A configurable transformer-based encoder that converts input text into embeddings. I choose some sentences as context and some to mask out.

2. **Predictor:** Given the encoder’s context embeddings, the predictor estimates the embeddings of the masked sentences. This is done in feature space, not token space.

3. **Conceptual Loss:** I use a smooth L1 or similar loss between the predictor’s output embeddings and the target embeddings for the masked sentences. No token reconstruction is involved.

## Comparison to I-JEPA

- **I-JEPA:** Predicts image patch embeddings from visible patches, never dealing with pixels directly.
- **LANG-JEPA:** Predicts sentence embeddings from visible sentences, never dealing directly with token reconstruction.

Both aim for semantic-level understanding. Instead of pixel-level or token-level detail, the model focuses on conceptual coherence.

## File Structure and Explanation

The repository is inspired by I-JEPA’s structure but adapted for text:

```python
LANG-JEPA/
├── configs/
│   ├── base_lang_config.yaml
│   ├── ... (other YAML configs)
│   └── README.md           # Document configuration usage
│
├── main.py                 # Entrypoint for single-machine training
├── README.md               # This file, explaining the project
│
└── src/
    ├── datasets/
    │   ├── fineweb_edu.py              # Analogous to imagenet1k.py in I-JEPA
    │   │                                # Loads text data from FineWeb-Edu dataset
    │   │                                # by streaming from Hugging Face datasets.
    │   │                                # Returns raw text samples.
    │   │
    │   └── utils/
    │       └── sentence_splitting.py   # Provides SentenceSplitter and SentenceSplitterConfig
    │                                    # Splits raw text into sentences for masking.
    │
    ├── masks/
    │   ├── lang_mask_collator.py        # Handles tokenization and sentence-level masking.
    │   │                                # It mimics I-JEPA’s image collators, but now:
    │   │                                #  - Splits text into sentences
    │   │                                #  - Chooses some sentences to mask
    │   │                                #  - Tokenizes and prepares input_ids and attention_masks
    │   │                                #  - Produces enc_masks and pred_masks (lists of token indices for context/prediction)
    │
    ├── models/
    │   └── text_transformer.py          # A single transformer file analogous to vision_transformer.py in I-JEPA.
    │                                    #  - Defines TextTransformer: a configurable transformer encoder
    │                                    #    that can scale in size (#layers, #heads, embed_dim, etc.)
    │                                    #  - Defines TextPredictor: a simple MLP that maps context embeddings to
    │                                    #    predicted embeddings for masked sentences.
    │                                    #  - build_text_transformer(): function to create encoder and predictor.
    │
    ├── utils/
    │   ├── logging.py                   # Utilities for logging metrics, CSVLogger, AverageMeter, etc.
    │   ├── schedulers.py                # WarmupCosineSchedule, CosineWDSchedule for LR and WD schedules.
    │   ├── tensors.py                   # Tensor utilities (e.g., trunc_normal_).
    │   └── __init__.py
    │
    ├── transforms/
    │   └── text_transforms.py           # If any text-level augmentations or transforms are needed, they go here.
    │
    ├── helper.py                        # Functions to:
    │                                    #  - init_model(): build encoder & predictor
    │                                    #  - init_optimizer(): create optimizer & schedulers
    │                                    #  - load_checkpoint() & save_checkpoint(): model checkpointing
    │
    ├── train.py                         # Main training loop:
    │                                    #  - Loads dataset (TextDataset from fineweb_edu.py)
    │                                    #  - Uses LangMaskCollator to produce masked inputs
    │                                    #  - Initializes models & optimizer using helper.py
    │                                    #  - Runs the forward pass: encoder for target features, encoder+predictor for masked predictions
    │                                    #  - Computes loss in concept space and updates parameters
    │
    └── __init__.py
```
### How It Relates to I-JEPA’s Structure

- **fineweb_edu.py** ↔ **imagenet1k.py** in I-JEPA: Loads the dataset.
- **lang_mask_collator.py** ↔ **masks/default.py / multiblock.py** in I-JEPA: Handles masking logic.
- **text_transformer.py** ↔ **vision_transformer.py** in I-JEPA: The main model architecture.
- **train.py** ↔ The main training script in I-JEPA.
- **helper.py**, **utils**, and **configs** follow the same pattern.

## Summary

LANG-JEPA is my attempt to take the principles of I-JEPA—predicting in feature space rather than output space—and apply them to language. By masking entire sentences and predicting conceptual embeddings rather than tokens, I encourage the model to form a deeper, more conceptual internal representation of language.