import logging
from typing import Callable, List

import torch
from torch.utils.data import Dataset


class _TextDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: Callable,
        max_length: int = 128,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def _get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Returns a configured logger.

    Args:
        name (str): The name of the logger (typically the module name).
        level (int): The logging level (e.g., logging.DEBUG, logging.INFO).

    Returns:
        logging.Logger: The configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    ch = logging.StreamHandler()
    ch.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)

    logger.addHandler(ch)

    return logger
