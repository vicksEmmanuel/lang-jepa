from dataclasses import dataclass

from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer

from src.common.datasets.utils.sentence_splitting import (
    SentenceSplitter,
    SentenceSplitterConfig,
)


@dataclass
class Sentence:
    text: str
    start_idx: int
    end_idx: int


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
        window_size: int = 25,
        min_sentences: int = 2,
        tokenizer: PreTrainedTokenizer | None = None,
        max_tokens: int | None = None,
        cache_dir: str = "~/.cache/huggingface/datasets",
    ):
        """Enhanced dataset wrapper with precise sentence boundary handling.

        Args:
            train_file: Which dataset file to load
            limit: Number of documents to process
            min_length: Minimum text length to consider
            window_size: Number of sentences to use as context (default: 25)
            min_sentences: Minimum sentences required (default: 2)
            tokenizer: Optional tokenizer for length checking
            max_tokens: Optional maximum tokens per context window
            cache_dir: HuggingFace cache directory
        """
        self.samples: list[DatasetOutput] = []
        self.stats = {
            "total_docs": 0,
            "docs_processed": 0,
            "docs_rejected_length": 0,
            "docs_rejected_sentences": 0,
            "context_target_pairs": 0,
            "pairs_rejected_length": 0,
        }

        # Load dataset
        print(f"Loading dataset with {window_size}-sentence sliding window...")
        # ds = load_dataset(
        #     path="HuggingFaceFW/fineweb-edu",
        #     name=train_file,
        #     split="train",
        #     streaming=True,
        #     cache_dir=cache_dir,
        # )

        import pandas as pd
        train = pd.read_parquet('/mnt/c/Users/kodeb/OneDrive/Desktop/vicks/Ai/lang-jepa/src/common/datasets/data/train.parquet')

        from datasets import Dataset, DatasetDict
        train_dataset = Dataset.from_pandas(train)

        def preprocess_function(examples):
            prompt = examples["prompt"]
            response_a = examples["response_a"]
            response_b = examples["response_b"]
            language = examples["language"]
            answer = examples["winner"]

            prompt_id = "<prompt>"
            response_1_id = "<response_1_id>"
            response_2_id = "<response_2_id>"
            answer_id = "<answer>"


            result = []

            for i in range(len(prompt)):

                output = "Response 1" if answer[i] == "model_a" else "Response 2"
                answer_preamble = "Final Decision: Based on above analysis, the user's preferences is"
                output_labels = "Answer: "


                cot_steps = [
                    f"""
                    Step 1: Language Analysis
                    - Identify prompt language: {language[i]}
                    - Verify response language consistency
                    - Check language appropriateness
                    """
                    f"""Step 2 : Content Quality Assessment
                    - Check response relevance to prompt
                    - Compare comprehensiveness
                    - Evaluate clarity and naturalness
                    """,
                ]
                
                steps_text = "\n".join(cot_steps)

                cot_prompt = f"Given the prompt {prompt_id} `{prompt[i]}` \n Response 1 : {response_1_id}`{response_a[i]}` \n Respone 2 : {response_2_id}`{response_b[i]}` " + "\n" + steps_text + "\n" + answer_preamble + answer_id + output_labels + output
                result.append(cot_prompt)

            
            return {
                "text": result
            }



        ds = train_dataset.map(
            (lambda x: preprocess_function(
                x, 
            )),
            batched=True,
        )

        for i in range(5):
            print("Input: ", ds["text"][i], "\n")
            print(f"{'='*100}\n")


        # Initialize sentence splitter
        splitter = SentenceSplitter(SentenceSplitterConfig())

        # Process documents
        pbar = tqdm(total=limit, desc="Processing documents", unit="docs")

        for doc in ds:
            self.stats["total_docs"] += 1
            text = doc.get("text", "").strip()

            # Check minimum length
            if len(text) < min_length:
                self.stats["docs_rejected_length"] += 1
                continue

            try:
                # Split into sentences
                sentences = splitter([text])[0]
                if len(sentences) < min_sentences:
                    self.stats["docs_rejected_sentences"] += 1
                    continue

                # Find sentence boundaries in original text
                sentence_objs: list[Sentence] = []
                search_start = 0

                for sent in sentences:
                    # Find the sentence in the original text
                    start_idx = text.index(sent, search_start)
                    end_idx = start_idx + len(sent)

                    sentence_objs.append(
                        Sentence(text=sent, start_idx=start_idx, end_idx=end_idx)
                    )
                    search_start = end_idx

                # Create context-target pairs with sliding window
                for i in range(1, len(sentence_objs)):
                    # Get previous sentences as context (up to window_size)
                    start_sent_idx = max(0, i - window_size)

                    # Get exact text slice from original document
                    context_start = sentence_objs[start_sent_idx].start_idx
                    context_end = sentence_objs[i - 1].end_idx
                    context = text[context_start:context_end]

                    # Get target sentence with exact boundaries
                    target = text[sentence_objs[i].start_idx : sentence_objs[i].end_idx]

                    # Check token length if tokenizer provided
                    if tokenizer and max_tokens:
                        context_tokens = len(tokenizer.encode(context))
                        if context_tokens > max_tokens:
                            self.stats["pairs_rejected_length"] += 1
                            continue

                    self.samples.append(
                        DatasetOutput(
                            context=context,
                            target=target,
                        )
                    )
                    self.stats["context_target_pairs"] += 1

                self.stats["docs_processed"] += 1
                pbar.update(1)

                if limit and self.stats["docs_processed"] >= limit:
                    break

            except Exception as e:
                print(f"Error processing document: {e}")
                continue

        pbar.close()

        # Print statistics
        print("\nDataset Processing Statistics:")
        print(f"Total documents seen: {self.stats['total_docs']:,}")
        print(f"Documents processed: {self.stats['docs_processed']:,}")
        print(f"Documents rejected (length): {self.stats['docs_rejected_length']:,}")
        print(
            f"Documents rejected (sentences): {self.stats['docs_rejected_sentences']:,}"
        )
        print(f"Context-target pairs generated: {self.stats['context_target_pairs']:,}")
        print(f"Pairs rejected (length): {self.stats['pairs_rejected_length']:,}")

        if not self.samples:
            raise RuntimeError(
                f"No valid samples found in dataset ({train_file}). "
                f"Try adjusting the minimum length ({min_length}) or "
                f"minimum sentences ({min_sentences}) requirements."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> DatasetOutput:
        return self.samples[idx]


def worker_init_fn(worker_id: int) -> None:
    """Initialize any worker-specific resources."""
    # No need for worker-specific initialization anymore since we process
    # everything in __init__
    pass
