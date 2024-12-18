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
./
├── src
│   ├── common
│   │   ├── datasets
│   │   │   ├── utils
│   │   │   │   └── sentence_splitting.py     # Sentence splitting utilities
│   │   │   └── fineweb_edu.py                # FineWeb-Edu dataset wrapper
│   │   ├── config.py                         # Configuration classes (Pydantic-based)
│   │   ├── logging.py                        # Logging utilities (CSV logging, meters)
│   │   └── schedulers.py                     # Learning rate and weight decay schedulers
│   │
│   ├── decoder
│   │   ├── configs
│   │   │   └── decoder_config.yaml           # YAML config for the decoder model and training
│   │   ├── utils
│   │   │   └── evaluation.py                 # Metrics (BLEU, ROUGE, etc.) and evaluation logic
│   │   ├── decoder_dataset.py                # Dataset and DataLoader utilities for decoder training
│   │   ├── models.py                         # Concept decoder model implementation
│   │   └── train.py                          # Decoder training loop
│   │
│   └── encoder
│       ├── configs
│       │   └── base_lang_config.yaml          # YAML config for the encoder model and training
│       ├── utils
│       │   ├── helper.py                      # Initialization and optimizer utilities
│       │   └── monitor.py                     # Monitoring and logging of training examples
│       ├── mask_collator.py                   # Masking logic: how sentences are masked at input
│       ├── models.py                          # LANG-JEPA encoder and predictor modules
│       └── train.py                           # Encoder (LANG-JEPA) training loop
│
├── main_decoder.py                            # Entrypoint for decoder training
├── main_encoder.py                            # Entrypoint for encoder (LANG-JEPA) training
├── pyproject.toml                             # Dependencies and project configuration
└── README.md                                  # This readme
```

## Configuration
### Encoder Configuration
Defined in `src/encoder/configs/base_lang_config.yaml`.

Controls aspects like:
- Model architecture (layers, heads, dimensions)
- Data loading and masking ratio
- Optimization parameters (learning rate, epochs, warmup, weight decay)
- Logging directories and frequencies

### Decoder Configuration
Defined in `src/decoder/configs/decoder_config.yaml`.

Controls the concept decoder model size, training hyperparameters, and evaluation steps.


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
   poetry shell
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