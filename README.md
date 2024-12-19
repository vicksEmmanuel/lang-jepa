# LANG-JEPA: Learning to Think in Latent Space

LANG-JEPA is an experimental language model architecture that operates in "concept space" rather than "token space." Building on Meta AI's JEPA framework, it predicts semantic features of future text rather than raw tokens, focusing on conceptual understanding and semantic relationships.

Previous JEPA implementations include:
- [I-JEPA](https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/) for images: Predicts feature representations of masked image regions
- [V-JEPA](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/) for videos: Predicts future visual features without pixel reconstruction

LANG-JEPA applies this approach to text, training models to predict feature-level representations of future text segments rather than specific tokens. The goal is to develop models that reason at a conceptual level, like humans.

## How It Works

LANG-JEPA learns by predicting the semantic features of upcoming text. Given a sequence of text, it:
1. Encodes both the context and the next sentence into a semantic latent space
2. Learns to predict the latent representation of the next sentence from the context
3. Uses cosine similarity in the latent space as a training signal

The system consists of two core components:

### 1. LANG-JEPA Encoder 
- A transformer-based model that transforms text into semantic embeddings
- Projects input text into a high-dimensional latent space
- Learns to capture semantic relationships between sentences

### 2. Concept Decoder
- Converts learned feature embeddings back into human-readable text
- Enables evaluation and interpretation of the model's semantic understanding
- Trained separately after the encoder

## Architecture

### Encoder Architecture:
- Text Encoder: Transforms input into semantic embeddings
- Context Processing: Processes context sequence with self-attention
- Feature Prediction: Uses attention to predict next sentence embeddings
- Loss: Cosine similarity between predicted and actual next sentence embeddings

### Decoder Architecture:
- Projects LANG-JEPA embeddings to decoder space
- Generates text via transformer decoder
- Trains with teacher forcing and cross-entropy loss
- Evaluates using reconstruction metrics

## File Structure

```
./
├── src
│   ├── common
│   │   ├── datasets
│   │   │   ├── utils
│   │   │   │   └── sentence_splitting.py     # Sentence splitting utilities
│   │   │   └── fineweb_edu.py               # FineWeb-Edu dataset wrapper
│   │   ├── config.py                        # Configuration classes (Pydantic-based)
│   │   ├── logging.py                       # Logging utilities (CSV logging, meters)
│   │   └── schedulers.py                    # Learning rate and weight decay schedulers
│   │
│   ├── decoder
│   │   ├── configs
│   │   │   └── decoder_config.yaml          # YAML config for the decoder model
│   │   ├── utils
│   │   │   └── evaluation.py                # Metrics (BLEU, ROUGE, etc.)
│   │   ├── decoder_dataset.py               # Dataset utilities for decoder
│   │   ├── models.py                        # Concept decoder model
│   │   └── train.py                         # Decoder training loop
│   │
│   └── encoder
│       ├── configs
│       │   └── base_lang_config.yaml        # YAML config for the encoder model
│       ├── utils
│       │   ├── helper.py                    # Initialization utilities
│       │   └── monitor.py                   # Training monitoring and logging
│       ├── collator.py                      # Dataset collation for training
│       ├── models.py                        # LANG-JEPA encoder and predictor
│       └── train.py                         # Encoder training loop
│
├── main_decoder.py                          # Decoder training entry point
├── main_encoder.py                          # Encoder training entry point
├── pyproject.toml                           # Dependencies and configuration
└── README.md                                # This readme
```

## Configuration
### Encoder Configuration
Defined in `src/encoder/configs/base_lang_config.yaml`.

Controls:
- Model architecture (layers, heads, dimensions)
- Data loading and sequence length
- Optimization parameters (learning rate, epochs, warmup)
- Logging settings

### Decoder Configuration
Defined in `src/decoder/configs/decoder_config.yaml`.

Controls:
- Decoder model architecture
- Training hyperparameters
- Evaluation settings

## Training Process

1. **LANG-JEPA Training:**
   ```
   Text → Split into Context/Target → Encode → Predict Next Features → Update Model
   ```

2. **Decoder Training:**
   ```
   Concept → Project → Generate Text → Compare with Original → Update Decoder
   ```

3. **Evaluation:**
   - Feature similarity in latent space
   - BLEU and ROUGE scores for generated text
   - Perplexity for language model quality
   - Semantic similarity metrics

## Getting Started

1. Install dependencies:
   ```bash
   poetry shell
   poetry install
   ```

2. Train LANG-JEPA encoder:
   ```bash
   python main_encoder.py
   ```

3. Train decoder (optional, for text generation):
   ```bash
   python main_decoder.py
   ```

## Model Details

### Encoder Architecture
- Built on top of any transformer model (RoBERTa, GPT2, etc.)
- Customized for semantic feature prediction
- Outputs normalized embeddings in latent space

### Training Objectives
- Primary: Next sentence feature prediction
- Loss: Cosine similarity in normalized latent space
- Regularization: Weight decay with cosine schedule

### Key Features
- Works directly in semantic space
- No token-level predictions
- Focus on semantic relationships
- Efficient training with cosine similarity