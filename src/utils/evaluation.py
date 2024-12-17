from dataclasses import dataclass

import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from rouge_score import rouge_scorer
from torch import Tensor
from transformers import PreTrainedTokenizer


@dataclass
class DecoderMetrics:
    """Holds evaluation metrics for concept decoder."""

    bleu: float
    rouge: dict[str, float]
    perplexity: float
    concept_cosine_sim: float
    diversity: float


class ConceptMetrics:
    """Evaluates concept decoder performance."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
    ):
        self.tokenizer = tokenizer
        self.device = device
        self.rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"])

    @torch.no_grad()
    def compute_metrics(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        original_texts: list[str],
        generated_texts: list[str],
        concept_embeddings: Tensor | None = None,
    ) -> DecoderMetrics:
        """
        Compute all metrics for generated text and concepts.

        Args:
            encoder: LANG-JEPA encoder model
            decoder: Concept decoder model
            original_texts: Ground truth texts
            generated_texts: Generated texts from decoder
            concept_embeddings: Optional pre-computed concept embeddings

        Returns:
            DecoderMetrics containing all computed metrics
        """
        # Compute BLEU score
        refs = [[t.split()] for t in original_texts]  # Split into words
        hyps = [t.split() for t in generated_texts]
        bleu = corpus_bleu(refs, hyps)

        # Compute ROUGE scores
        rouge_scores = {
            name: sum(
                self.rouge.score(orig, gen)[name].fmeasure
                for orig, gen in zip(original_texts, generated_texts, strict=False)
            )
            / len(original_texts)
            for name in ["rouge1", "rouge2", "rougeL"]
        }

        # Compute perplexity
        perplexity = self._compute_perplexity(decoder, original_texts)

        # Compute concept similarity if embeddings not provided
        if concept_embeddings is None:
            orig_concepts = encoder(
                self.tokenizer(
                    original_texts, return_tensors="pt", padding=True, truncation=True
                ).to(self.device)
            )
        else:
            orig_concepts = concept_embeddings

        gen_concepts = encoder(
            self.tokenizer(
                generated_texts, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
        )

        concept_sim = (
            F.cosine_similarity(orig_concepts.mean(dim=1), gen_concepts.mean(dim=1))
            .mean()
            .item()
        )

        # Compute diversity
        diversity = self._compute_diversity(generated_texts)

        return DecoderMetrics(
            bleu=bleu,
            rouge=rouge_scores,
            perplexity=perplexity,
            concept_cosine_sim=concept_sim,
            diversity=diversity,
        )

    def _compute_perplexity(
        self,
        decoder: torch.nn.Module,
        texts: list[str],
    ) -> float:
        """Compute perplexity of generated texts."""
        # Tokenize texts
        encodings = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        input_ids = encodings["input_ids"]

        # Forward pass through decoder
        with torch.no_grad():
            outputs = decoder(input_ids=input_ids[:, :-1], labels=input_ids[:, 1:])

        return torch.exp(outputs.loss).item()

    def _compute_diversity(self, texts: list[str]) -> float:
        """Compute lexical diversity of generated texts."""
        if not texts:
            return 0.0

        # Split into words and get unique words
        all_words = []
        for text in texts:
            all_words.extend(text.split())

        if not all_words:
            return 0.0

        unique_words = set(all_words)
        return len(unique_words) / len(all_words)


class SampleGenerator:
    """Generates and displays sample decoder outputs."""

    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def generate_samples(
        self, texts: list[str], num_samples: int = 3
    ) -> list[dict[str, str]]:
        """Generate samples for visualization."""
        samples = []

        for text in texts[:num_samples]:
            # Encode to concept space
            inputs = self.tokenizer(
                text, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)

            concept = self.encoder(inputs["input_ids"])

            # Generate from concept
            generated = self.decoder.generate(concept, self.tokenizer)[
                0
            ]  # Take first (and only) generation

            samples.append(
                {
                    "original": text,
                    "generated": generated,
                    "bleu": sentence_bleu([text.split()], generated.split()),
                }
            )

        return samples


def format_metrics(metrics: DecoderMetrics) -> str:
    """Format metrics for printing."""
    return (
        f"BLEU: {metrics.bleu:.4f}\n"
        f"ROUGE-1: {metrics.rouge['rouge1']:.4f}\n"
        f"ROUGE-2: {metrics.rouge['rouge2']:.4f}\n"
        f"ROUGE-L: {metrics.rouge['rougeL']:.4f}\n"
        f"Perplexity: {metrics.perplexity:.2f}\n"
        f"Concept Similarity: {metrics.concept_cosine_sim:.4f}\n"
        f"Diversity: {metrics.diversity:.4f}"
    )
