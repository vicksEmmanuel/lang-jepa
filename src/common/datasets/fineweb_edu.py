import time

from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm.auto import tqdm


class TextDataset(Dataset):
    def __init__(
        self,
        *,
        train_file: str,
        limit: int | None,
        min_length: int,
        cache_dir: str = "~/.cache/huggingface/datasets",
    ):
        """
        A dataset wrapper for FineWeb-Edu data with progress bar.

        Args:
            train_file (str): Which subset of FineWeb-Edu to load.
                              For example: "CC-MAIN-2024-10" or "sample-10BT".
            limit (int): Number of documents to load from the streaming dataset.
            min_length (int): Minimum text length to consider a sample valid.
        """
        total_start = time.time()

        # Initialize metrics
        self.samples = []
        self.stats = {
            "dataset_load_time": 0,
            "processing_time": 0,
            "total_docs_processed": 0,
            "docs_accepted": 0,
            "docs_rejected": 0,
        }

        # Time the dataset loading
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

        # Try to get dataset info (this won't download the full dataset)
        try:
            info = ds._info
            if hasattr(info, "splits") and hasattr(info.splits, "total_num_examples"):
                total_docs = info.splits.total_num_examples
                print(f"Total documents in dataset: {total_docs:,}")
            else:
                total_docs = None
                print("Total document count not available in dataset metadata")
        except:
            total_docs = None
            print("Could not retrieve dataset size information")

        # Process documents with progress bar
        processing_start = time.time()
        count = 0

        # Create progress bar
        pbar = tqdm(
            total=limit
            if limit
            else None,  # Use limit if specified, otherwise unknown total
            desc="Processing documents",
            unit="docs",
            dynamic_ncols=True,  # Automatically adjust to terminal width
        )

        for doc in ds:
            self.stats["total_docs_processed"] += 1

            text = doc.get("text", "").strip()
            if len(text) >= min_length:
                self.samples.append(text)
                self.stats["docs_accepted"] += 1
                count += 1
                pbar.update(1)
                if count >= limit:
                    break
            else:
                self.stats["docs_rejected"] += 1

        pbar.close()
        self.stats["processing_time"] = time.time() - processing_start
        total_time = time.time() - total_start

        if not self.samples:
            raise RuntimeError(
                f"No samples found in FineWeb-Edu ({train_file}) with min_length={min_length}. "
                "Try adjusting parameters or ensuring dataset is accessible."
            )

        # Print detailed timing report
        print("\nDataset Loading Complete:")
        print(f"- Dataset load time: {self.stats['dataset_load_time']:.2f}s")
        print(f"- Document processing time: {self.stats['processing_time']:.2f}s")
        print(f"- Total time: {total_time:.2f}s")
        print("\nDocument Statistics:")
        print(f"- Total documents processed: {self.stats['total_docs_processed']:,}")
        print(f"- Documents accepted: {self.stats['docs_accepted']:,}")
        print(f"- Documents rejected: {self.stats['docs_rejected']:,}")
        print(
            f"- Processing rate: {self.stats['total_docs_processed']/self.stats['processing_time']:.1f} docs/sec"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
