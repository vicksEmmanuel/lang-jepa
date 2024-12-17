# LANG-JEPA: Learning to Think in Concept Space

LANG-JEPA is an experimental language model architecture that operates in "concept space" rather than "token space." Building on Meta AI's JEPA framework, it predicts abstract features instead of raw tokens, focusing on conceptual understanding rather than exact word matching.

Previous JEPA implementations include:
- [I-JEPA](https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/) for images: Predicts feature representations of masked image regions
- [V-JEPA](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/) for videos: Predicts future visual features without pixel reconstruction

LANG-JEPA applies this approach to text, training models to predict feature-level representations of masked text segments rather than specific tokens. The goal is to develop models that reason at a conceptual level, like humans.

## How It Works

The system consists of two core components:

### 1. LANG-JEPA Encoder 
A transformer-based model that predicts embeddings of masked sentences from surrounding context, operating entirely in feature space.

### 2. Concept Decoder
Converts learned feature embeddings back into human-readable text to evaluate the model's conceptual understanding.

## Architecture

### Encoder:
- Text Encoder transforms input into embeddings
- Processes context and masked sentences
- Predicts masked sentence embeddings
- Uses Smooth L1 loss

### Decoder:
- Projects LANG-JEPA embeddings to decoder space
- Generates text via transformer decoder
- Trains with teacher forcing and cross-entropy loss
- Evaluates using multiple reconstruction metrics

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
   poetry shell && poetry install
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
