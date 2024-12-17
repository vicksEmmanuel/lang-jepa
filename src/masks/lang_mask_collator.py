import random

import torch

from src.datasets.utils.sentence_splitting import (
    SentenceSplitter,
    SentenceSplitterConfig,
)


class LangMaskCollator:
    def __init__(
        self,
        tokenizer,
        mask_ratio: float, # 0.3
        max_length: int, # 512
        seed: int = 42,
        sentence_split_config: SentenceSplitterConfig = None,
    ):
        """
        LANG-JEPA Collator for sentence-level masking using SentenceSplitter.

        Args:
            tokenizer: A Hugging Face-compatible tokenizer instance.
            mask_ratio (float): Proportion of sentences to mask.
            max_length (int): Maximum sequence length after tokenization & padding.
            seed (int): Random seed for reproducibility.
            sentence_split_config (SentenceSplitterConfig): Configuration for the SentenceSplitter.
        """
        self.tokenizer = tokenizer
        self.mask_ratio = mask_ratio
        self.max_length = max_length
        random.seed(seed)

        if sentence_split_config is None:
            sentence_split_config = SentenceSplitterConfig()

        self.sentence_splitter = SentenceSplitter(sentence_split_config)

    def __call__(
        self, batch: list[str]
    ) -> tuple[
        dict[str, torch.Tensor], list[list[torch.Tensor]], list[list[torch.Tensor]]
    ]:
        """
        Processes a batch of raw text samples into tokenized, masked inputs and corresponding mask indices.

        Args:
            batch (List[str]): A batch of raw text strings.

        Returns:
            input_dict (Dict[str, torch.Tensor]): Tokenized inputs (e.g. input_ids, attention_mask).
            enc_masks (List[List[torch.Tensor]]): For each sample, a list of tensors with token indices for context sentences.
            pred_masks (List[List[torch.Tensor]]): For each sample, a list of tensors with token indices for masked sentences.
        """
        # Step 1: Split texts into sentences using SentenceSplitter
        # This returns a list of lists, where each element corresponds to one text in the batch.
        batch_sentences = self.sentence_splitter(batch)

        # Step 2: Decide which sentences to mask for each sample
        masked_sent_indices = []
        for sentences in batch_sentences:
            num_sentences = len(sentences)
            # Ensure at least one sentence gets masked if possible
            num_to_mask = (
                max(1, int(self.mask_ratio * num_sentences)) if num_sentences > 0 else 0
            )
            if num_to_mask > 0:
                mask_indices = random.sample(range(num_sentences), k=num_to_mask)
            else:
                mask_indices = []
            masked_sent_indices.append(set(mask_indices))

        # Step 3: Tokenize, build sequences, and apply masking
        input_ids_batch = []
        attention_mask_batch = []
        enc_masks_batch = []
        pred_masks_batch = []

        for b_idx, sentences in enumerate(batch_sentences):
            if len(sentences) == 0:
                # Handle empty text case: just produce a minimal sequence
                full_ids = [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id]
                attention_mask = [1, 1]
                # Pad if needed
                if len(full_ids) < self.max_length:
                    padding_length = self.max_length - len(full_ids)
                    full_ids += [self.tokenizer.pad_token_id] * padding_length
                    attention_mask += [0] * padding_length

                input_ids_batch.append(full_ids)
                attention_mask_batch.append(attention_mask)
                enc_masks_batch.append([])
                pred_masks_batch.append([])
                continue

            # Tokenize all sentences individually
            tokenized_sents = self.tokenizer(
                sentences,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )["input_ids"]

            mask_set = masked_sent_indices[b_idx]
            masked_sequence = []
            sentence_token_indices = []
            current_token_idx = 0

            for s_idx, sent_ids in enumerate(tokenized_sents):
                start_idx = current_token_idx
                if s_idx in mask_set:
                    # Replace all tokens in this sentence with [MASK]
                    masked_sequence.extend(
                        [self.tokenizer.mask_token_id] * len(sent_ids)
                    )
                else:
                    masked_sequence.extend(sent_ids)
                end_idx = current_token_idx + len(sent_ids)
                sentence_token_indices.append((start_idx, end_idx))
                current_token_idx = end_idx

            # Add special tokens [CLS] and [SEP] if required
            full_ids = (
                [self.tokenizer.cls_token_id]
                + masked_sequence
                + [self.tokenizer.sep_token_id]
            )

            # Adjust indices due to the prepended [CLS]
            sentence_token_indices = [
                (start + 1, end + 1) for (start, end) in sentence_token_indices
            ]

            # Truncate if necessary
            if len(full_ids) > self.max_length:
                full_ids = full_ids[: self.max_length]
                truncated_indices = []
                for start, end in sentence_token_indices:
                    if start < self.max_length:
                        truncated_indices.append((start, min(end, self.max_length)))
                sentence_token_indices = truncated_indices

            # Pad if needed
            attention_mask = [1] * len(full_ids)
            while len(full_ids) < self.max_length:
                full_ids.append(self.tokenizer.pad_token_id)
                attention_mask.append(0)

            input_ids_batch.append(full_ids)
            attention_mask_batch.append(attention_mask)

            # Build enc_masks and pred_masks
            enc_masks = []
            pred_masks = []
            for s_idx, (start, end) in enumerate(sentence_token_indices):
                idx_tensor = torch.arange(start, end, dtype=torch.long)
                if s_idx in mask_set:
                    pred_masks.append(idx_tensor)
                else:
                    enc_masks.append(idx_tensor)

            enc_masks_batch.append(enc_masks)
            pred_masks_batch.append(pred_masks)

        # Convert to tensors
        input_ids = torch.tensor(input_ids_batch, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask_batch, dtype=torch.long)
        input_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        return input_dict, enc_masks_batch, pred_masks_batch
