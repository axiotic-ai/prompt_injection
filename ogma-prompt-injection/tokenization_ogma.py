"""OgmaTokenizerFast — wraps PreTrainedTokenizerFast, shifts token ids by
N_SPECIAL so they align with Ogma's embedding table.

Ogma reserved vocab ids (0-6):
  0 <pad>  1 <unk>  2 [CLS]  3 [SEP]  4 [MASK]  5 [DOC]  6 [SYM]
Regular SentencePiece tokens start at 7.

The tokenizer post-processor already adds [CLS] / [SEP] around every input.
This wrapper shifts ALL content positions (attention_mask == 1) up by
N_SPECIAL so that [CLS]->9, [SEP]->10, and content tokens land where the
model was trained to see them.  Padding positions (attention_mask == 0) stay
at 0 (Ogma pad id).
"""
from __future__ import annotations

import torch
from transformers import PreTrainedTokenizerFast
from transformers.tokenization_utils_base import BatchEncoding

__all__ = ["OgmaTokenizerFast"]

N_SPECIAL = 7


class OgmaTokenizerFast(PreTrainedTokenizerFast):
    N_SPECIAL = N_SPECIAL

    def _shift(self, ids, mask):
        if isinstance(ids, torch.Tensor):
            return ids + self.N_SPECIAL * mask.long()
        return [
            [i + self.N_SPECIAL if m else i for i, m in zip(row_i, row_m)]
            for row_i, row_m in zip(ids, mask)
        ]

    def __call__(self, *args, **kwargs) -> BatchEncoding:
        kwargs.setdefault("padding", True)
        kwargs.setdefault("truncation", True)
        kwargs.setdefault("max_length", self.model_max_length or 1024)
        enc = super().__call__(*args, **kwargs)
        if "input_ids" in enc and "attention_mask" in enc:
            enc["input_ids"] = self._shift(enc["input_ids"], enc["attention_mask"])
        return enc
