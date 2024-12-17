import os
from dataclasses import dataclass, field

import torch
from devtools import debug
from wtpsplit import SaT, indices_to_sentences

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class SentenceSplitterConfig:
    model_name: str = "sat-3l-sm"
    sentence_suffix: str = "_sentences"
    sentence_threshold: float = 0.01
    max_sentence_len: int = 256
    min_text_length: int = 10
    min_unique_chars: int = 0
    fallback_separators: list[str] = field(
        default_factory=lambda: [
            "...",
            "\n",
            "!",
            "?",
            ";",
            ":",
            ".",
            ",",
            "\t",
            " ",
        ]
    )
    device: str = "cuda"
    remove_whitespace_before_inference: bool = False
    batch_size: int = 256
    block_size: int = 256
    stride: int = 256
    outer_batch_size: int = 1024
    verbose: bool = False
    pad_last_batch: bool = False


class SentenceSplitter:
    def __init__(self, config: SentenceSplitterConfig):
        self.config = config
        device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        try:
            self.model = SaT(
                self.config.model_name,
                from_pretrained_kwargs={"local_files_only": True},
            )
        except Exception:
            self.model = SaT(self.config.model_name)

        if "cuda" in config.device and torch.cuda.is_available():
            self.model.half()

        self.model.eval().to(device)
        self.device = device

    @torch.inference_mode()
    def _resplit_long_sentence(self, sentence: str) -> list[str]:
        # If a single sentence is too long, split further by fallback separators
        # until no segment exceeds max_sentence_len or fallback is exhausted.
        segments = [sentence]
        for sep in self.config.fallback_separators:
            new_segments = []
            for seg in segments:
                if len(seg) > self.config.max_sentence_len:
                    # Split using the current separator
                    parts = [s.strip() for s in seg.split(sep) if s.strip()]
                    # If splitting didn't help (e.g., no sep found), just do a brute force word split.
                    if len(parts) == 1 and len(parts[0]) == len(seg):
                        parts = self._brute_force_split(seg)
                    new_segments.extend(parts)
                else:
                    new_segments.append(seg)
            segments = new_segments

        # Finally, ensure no segment is still too long
        final_segments = []
        for seg in segments:
            if len(seg) > self.config.max_sentence_len:
                final_segments.extend(self._brute_force_split(seg))
            else:
                final_segments.append(seg)

        return final_segments

    def _brute_force_split(self, text: str) -> list[str]:
        # Split by words if the text is still too large, ignoring separators
        words = text.split()
        chunks = []
        current_chunk = []

        for w in words:
            # +1 for space
            if (
                sum(len(x) + 1 for x in current_chunk) + len(w)
                > self.config.max_sentence_len
            ):
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [w]
            else:
                current_chunk.append(w)

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return [c.strip() for c in chunks if c.strip()]

    def _filter_by_unique_chars(self, sentences: list[str]) -> list[str]:
        if self.config.min_unique_chars <= 0:
            return sentences

        def unique_chars_count(s: str) -> int:
            return len(set(s))

        return [
            s for s in sentences if unique_chars_count(s) > self.config.min_unique_chars
        ]

    @torch.inference_mode()
    def __call__(self, texts: list[str]) -> list[list[str]]:
        # If single string, convert to list
        if isinstance(texts, str):
            texts = [texts]

        # Split texts using the model
        # Filter out too-short texts directly
        long_texts = [
            (i, t) for i, t in enumerate(texts) if len(t) > self.config.min_text_length
        ]
        short_texts = [
            (i, t) for i, t in enumerate(texts) if len(t) <= self.config.min_text_length
        ]

        # Extract the actual text for model inference
        long_text_strings = [t for _, t in long_texts]

        # Run the model
        outputs = self.model.split(
            long_text_strings,
            threshold=self.config.sentence_threshold,
            stride=self.config.stride,
            block_size=self.config.block_size,
            batch_size=self.config.batch_size,
            pad_last_batch=self.config.pad_last_batch,
            remove_whitespace_before_inference=self.config.remove_whitespace_before_inference,
            outer_batch_size=self.config.outer_batch_size,
            verbose=self.config.verbose,
        )

        # Now we have a list of sentence lists for each long text
        # Post-process each list:
        final_results = [None] * len(texts)

        # Insert short texts (they don't need splitting)
        for i, t in short_texts:
            final_results[i] = [t.strip()] if t.strip() else []

        # Process the long texts
        for (i, _), sentence_list in zip(long_texts, outputs, strict=False):
            # Strip sentences
            sentence_list = [s.strip() for s in sentence_list if s.strip()]

            # Resplit any long sentences
            resplit_sentences = []
            for sent in sentence_list:
                if len(sent) > self.config.max_sentence_len:
                    resplit_sentences.extend(self._resplit_long_sentence(sent))
                else:
                    resplit_sentences.append(sent)

            # Filter by unique chars if needed
            resplit_sentences = self._filter_by_unique_chars(resplit_sentences)

            final_results[i] = resplit_sentences

        return final_results


if __name__ == "__main__":
    import time

    config = SentenceSplitterConfig()
    splitter = SentenceSplitter(config)
    sample_texts = [
        "This is a test. It's a simple test, isn't it? Yes, it is!",
        "Short",
        "A very long sentence that definitely exceeds the maximum length of the sentence and should be split into multiple chunks by the splitter because it is too long to remain one single sentence yes very long isn't it wowowww this is super long ahuahiuhahaiuahu.",
    ]
    times = []
    for _ in range(10):
        start = time.time()
        result = splitter(sample_texts)
        debug(result)
        times.append(time.time() - start)
    print(f"Average time: {sum(times) / len(times)}, {times}")
