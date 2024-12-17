# LANG-JEPA

I am working on LANG-JEPA (Language-based Joint-Embedding Predictive Architecture), an experimental framework inspired by I-JEPA (Image-based Joint-Embedding Predictive Architecture) that aims to learn **conceptual embeddings** of text by predicting the latent representation of masked textual segments from visible context segments. Instead of focusing on generating or predicting tokens directly, I optimize the model in a "concept" space—feature vectors that represent underlying semantic meaning—before mapping them to words.

## What is LANG-JEPA?

I-JEPA is a self-supervised learning method originally designed for images. It predicts representations of unseen image patches from visible patches, working entirely in feature space without relying on pixel-level reconstructions. This leads to semantic, abstraction-rich representations.

I am taking the same approach for text. Instead of predicting tokens (like a masked language model), I mask entire sentences and have the model predict their conceptual embeddings from the visible sentences. By operating in concept space rather than token space, I hope the model will develop internal "conceptual maps" of meaning, less tied to any specific sequence of words. The goal is for deductive or conceptual reasoning circuits to emerge naturally, since humans often think in concepts first and then find words to express them.

## Key Components

### 1. LANG-JEPA Encoder 
- Takes input text and masks certain sentences
- Learns to predict embeddings of masked sentences from context
- Works in conceptual space rather than token space

### 2. Concept Decoder
- Takes learned conceptual embeddings and decodes them back to text
- Helps evaluate and interpret the learned concept space 
- Consists of:
  - Concept projection layer to map embeddings to decoder space
  - Transformer decoder to generate text from concepts
  - Evaluation metrics (BLEU, ROUGE, perplexity, etc.)

## Architecture Overview

### Encoder Architecture:
1. **Text Encoder:** Transformer-based encoder that converts input text into embeddings
2. **Context Processing:** Some sentences chosen as context, others masked
3. **Prediction:** Predicts embeddings of masked sentences from context embeddings
4. **Loss:** Smooth L1 loss between predicted and actual embeddings

### Decoder Architecture:
1. **Concept Projection:** Maps LANG-JEPA concept embeddings to decoder space
2. **Transformer Decoder:** Generates text from projected concepts
3. **Training:** Uses teacher forcing with cross-entropy loss
4. **Evaluation:** Measures reconstruction quality with multiple metrics

## File Structure

```
LANG-JEPA/
├── configs/
│   ├── base_lang_config.yaml     # Base configuration for LANG-JEPA
│   ├── decoder_config.yaml       # Configuration for concept decoder
│   └── README.md                 # Configuration documentation
│
├── main.py                       # Entrypoint for LANG-JEPA training
├── main_decoder.py              # Entrypoint for decoder training
├── README.md                    # This file
│
└── src/
    ├── datasets/
    │   ├── decoder_dataset.py    # Dataset and loaders for decoder training
    │   ├── fineweb_edu.py       # Main dataset loader
    │   └── utils/
    │       └── sentence_splitting.py
    │
    ├── masks/
    │   └── lang_mask_collator.py # Handles sentence-level masking
    │
    ├── models/
    │   ├── concept_decoder.py    # Decoder model architecture
    │   └── text_transformer.py   # LANG-JEPA encoder architecture
    │
    ├── utils/
    │   ├── evaluation.py         # Metrics for decoder evaluation
    │   ├── logging.py           
    │   ├── schedulers.py        
    │   └── tensors.py           
    │
    ├── train.py                  # LANG-JEPA training loop
    ├── train_decoder.py          # Decoder training loop
    └── helper.py                # Shared utilities
```

## Training Process

1. **LANG-JEPA Training:**
   ```
   Text → Split Sentences → Mask Some → Predict Embeddings → Update Model
   ```

2. **Decoder Training:**
   ```
   Concept → Project → Generate Text → Compare with Original → Update Decoder
   ```

3. **Evaluation:**
   - BLEU and ROUGE scores for text similarity
   - Perplexity for language model quality
   - Concept similarity between original and generated text
   - Lexical diversity metrics

## Getting Started

1. Install dependencies:
   ```bash
   poetry install
   ```

2. Train LANG-JEPA:
   ```bash
   python main.py
   ```

3. Train decoder:
   ```bash
   python main_decoder.py
   ```

## Configuration

- `base_lang_config.yaml`: Controls LANG-JEPA model size, training, and masking
- `decoder_config.yaml`: Sets up decoder architecture and training parameters

## Why This Matters

By training LANG-JEPA this way and adding a concept decoder, we can:
1. Verify the quality of learned conceptual representations
2. Make the model's behavior more interpretable
3. Understand how well it captures semantic meaning
4. Bridge between concept space and natural language

The decoder serves as both an evaluation tool and a window into how the model understands language at a conceptual level.