import time
from dataclasses import dataclass

from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from src.common.datasets.utils.sentence_splitting import (
    SentenceSplitter,
    SentenceSplitterConfig,
)


@dataclass
class DatasetOutput:
    context: str
    target: str


class TextDataset(Dataset):
    def __init__(
        self,
        *,
        train_file: str,
        limit: int | None,
        min_length: int,
        min_sentences: int = 2,
        cache_dir: str = "~/.cache/huggingface/datasets",
    ):
        """A dataset wrapper for FineWeb-Edu that splits text into context/target pairs.

        Args:
            train_file: Which subset of FineWeb-Edu to load (e.g., "CC-MAIN-2024-10")
            limit: Number of documents to load from streaming dataset
            min_length: Minimum text length to consider a sample valid
            min_sentences: Minimum number of sentences required (default: 2)
            cache_dir: Directory for caching HuggingFace datasets
        """
        total_start = time.time()

        # Initialize metrics
        self.samples: list[DatasetOutput] = []
        self.stats = {
            "dataset_load_time": 0,
            "processing_time": 0,
            "total_docs_processed": 0,
            "docs_accepted": 0,
            "docs_rejected_length": 0,
            "docs_rejected_sentences": 0,
            "split_failed": 0,
        }

        # Store configuration
        self.min_length = min_length
        self.min_sentences = min_sentences

        # Load dataset
        print("Loading dataset...")
        load_start = time.time()
        ds = load_dataset(
            path="HuggingFaceFW/fineweb-edu",
            name=train_file,
            split="train",
            streaming=True,
            cache_dir=cache_dir,
        )
        self.stats["dataset_load_time"] = time.time() - load_start

        # Process documents
        processing_start = time.time()
        count = 0

        # Create a sentence splitter for initial processing
        splitter = SentenceSplitter(SentenceSplitterConfig())

        pbar = tqdm(
            total=limit if limit else None,
            desc="Processing documents",
            unit="docs",
            dynamic_ncols=True,
        )

        for doc in ds:
            self.stats["total_docs_processed"] += 1
            text = doc.get("text", "").strip()

            # Check minimum length
            if len(text) < min_length:
                self.stats["docs_rejected_length"] += 1
                continue

            # Try to split into sentences
            try:
                sentences = splitter([text])[0]
                if len(sentences) < min_sentences:
                    self.stats["docs_rejected_sentences"] += 1
                    continue

                # Find and split at last sentence
                split_idx = text.rindex(sentences[-1])
                self.samples.append(
                    DatasetOutput(
                        context=text[:split_idx],
                        target=text[split_idx:],
                    )
                )
                self.stats["docs_accepted"] += 1
                count += 1
                pbar.update(1)

                if limit and count >= limit:
                    break

            except Exception:
                self.stats["split_failed"] += 1

        pbar.close()
        self.stats["processing_time"] = time.time() - processing_start
        total_time = time.time() - total_start

        if not self.samples:
            raise RuntimeError(
                f"No valid samples found in FineWeb-Edu ({train_file}). "
                f"Processed {self.stats['total_docs_processed']:,} documents:\n"
                f"- Rejected (length < {min_length}): {self.stats['docs_rejected_length']:,}\n"
                f"- Rejected (sentences < {min_sentences}): {self.stats['docs_rejected_sentences']:,}\n"
                f"- Split failed: {self.stats['split_failed']:,}"
            )

        # Print detailed timing report
        print("\nDataset Loading Complete:")
        print(f"- Dataset load time: {self.stats['dataset_load_time']:.2f}s")
        print(f"- Document processing time: {self.stats['processing_time']:.2f}s")
        print(f"- Total time: {total_time:.2f}s")
        print("\nDocument Statistics:")
        print(f"- Total documents processed: {self.stats['total_docs_processed']:,}")
        print(f"- Documents accepted: {self.stats['docs_accepted']:,}")
        print(f"- Rejected (length): {self.stats['docs_rejected_length']:,}")
        print(f"- Rejected (sentences): {self.stats['docs_rejected_sentences']:,}")
        print(f"- Split failed: {self.stats['split_failed']:,}")
        print(
            f"- Processing rate: {self.stats['total_docs_processed']/self.stats['processing_time']:.1f} docs/sec"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> DatasetOutput:
        """Get a sample at the given index.

        Returns:
            DatasetOutput containing:
                - context: Text before the last sentence
                - target: The last sentence
        """
        return self.samples[idx]


def worker_init_fn(worker_id: int) -> None:
    """Initialize any worker-specific resources."""
    # No need for worker-specific initialization anymore since we process
    # everything in __init__
    pass
