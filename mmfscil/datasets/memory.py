from typing import Dict

import torch
from torch.utils.data import Dataset


class MemoryDataset(Dataset):
    """MemoryDataset is a dataset that loads features
    """

    def __init__(
            self,
            feats: torch.Tensor,
            labels: torch.Tensor,
    ):
        self.feats = feats
        self.labels = labels
        assert len(self.feats) == len(self.labels), "The features and labels are with different sizes."

    def __len__(self) -> int:
        """Return length of the dataset."""
        return len(self.feats)

    def __getitem__(self, idx: int) -> Dict:
        return {"feat": self.feats[idx], "gt_label": self.labels[idx]}
