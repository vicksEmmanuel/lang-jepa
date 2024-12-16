from torch.utils.data import Dataset

from datasets import load_dataset


class TextDataset(Dataset):
    def __init__(
        self,
        *,
        train_file: str = "sample-10BT",
        limit: int = 10_000,
        min_length: int = 10,
        cache_dir: str = "~/.cache/huggingface/datasets",
    ):
        """
        A dataset wrapper for FineWeb-Edu data.

        Args:
            train_file (str): Which subset of FineWeb-Edu to load.
                              For example: "CC-MAIN-2024-10" or "sample-10BT".
            limit (int): Number of documents to load from the streaming dataset.
            min_length (int): Minimum text length to consider a sample valid.
        """
        self.samples = []
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name=train_file,
            split="train",
            streaming=True,
            cache_dir=cache_dir,
        )

        count = 0
        for doc in ds:
            text = doc.get("text", "").strip()
            if len(text) >= min_length:
                self.samples.append(text)
                count += 1
                if count >= limit:
                    break

        if not self.samples:
            raise RuntimeError(
                f"No samples found in FineWeb-Edu ({train_file}) with min_length={min_length}. "
                "Try adjusting parameters or ensuring dataset is accessible."
            )

        print(f"Loaded {len(self.samples)} samples from FineWeb-Edu ({train_file}).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
