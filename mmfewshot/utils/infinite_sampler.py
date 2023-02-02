# Copyright (c) OpenMMLab. All rights reserved.
import itertools
import math
from typing import Iterable, Iterator, Optional

import numpy as np
import torch
from mmcv.runner import get_dist_info
from torch.utils.data.sampler import Sampler

from .dist_utils import sync_random_seed


class InfiniteSampler(Sampler):
    """Return a infinite stream of index.

    The length of sampler is set to the actual length of dataset, thus the
    length of dataloader is still determined by the dataset. The
    implementation logic is referred to
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/samplers/grouped_batch_sampler.py

    Args:
        dataset (Iterable): The dataset.
        seed (int): Random seed. Default: 0.
        shuffle (bool): Whether shuffle the dataset or not. Default: True.
    """  # noqa: W605

    def __init__(self,
                 dataset: Iterable,
                 seed: int = 0,
                 shuffle: bool = True) -> None:
        self.dataset = dataset
        # In distributed sampling, different ranks should sample
        # non-overlapped data in the dataset. Therefore, this function
        # is used to make sure that each rank shuffles the data indices
        # in the same order based on the same seed. Then different ranks
        # could use different indices to select non-overlapped data from the
        # same data list.
        self.seed = sync_random_seed(seed)
        self.shuffle = shuffle
        self.size = len(dataset)
        self.indices = self._indices()
        self.epoch = 0

    def _infinite_indices(self) -> Iterator:
        """Infinitely yield a sequence of indices."""
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        while True:
            if self.shuffle:
                yield from torch.randperm(self.size, generator=g).tolist()
            else:
                yield from torch.arange(self.size).tolist()

    def _indices(self) -> Iterator:
        """Slice the infinite indices by rank."""
        yield from itertools.islice(self._infinite_indices(), 0, None)

    def __iter__(self) -> Iterator:
        yield from self.indices

    def __len__(self) -> int:
        """Length of dataset."""
        # The length of sampler is set to the actual length of dataset, thus
        # the length of dataloader is still determined by the dataset.
        return self.size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class InfiniteGroupSampler(Sampler):
    """Similar to `InfiniteSampler`, but all indices in a batch should be in
    the same group of flag.

    The length of sampler is set to the actual length of dataset, thus the
    length of dataloader is still determined by the dataset. The
    implementation logic is referred to
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/samplers/grouped_batch_sampler.py

    Args:
        dataset (Iterable): The dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU. Default: 1.
        seed (int): Random seed. Default: 0.
        shuffle (bool): Whether shuffle the indices of a dummy `epoch`, it
            should be noted that `shuffle` can not guarantee that you can
            generate sequential indices because it need to ensure
            that all indices in a batch is in a group. Default: True.
    """  # noqa: W605

    def __init__(self,
                 dataset: Iterable,
                 samples_per_gpu: int = 1,
                 seed: int = 0,
                 shuffle: bool = True) -> None:
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        # In distributed sampling, different ranks should sample
        # non-overlapped data in the dataset. Therefore, this function
        # is used to make sure that each rank shuffles the data indices
        # in the same order based on the same seed. Then different ranks
        # could use different indices to select non-overlapped data from the
        # same data list.
        self.seed = sync_random_seed(seed)
        self.shuffle = shuffle

        assert hasattr(self.dataset, 'flag')
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)
        # buffer used to save indices of each group
        self.buffer_per_group = {k: [] for k in range(len(self.group_sizes))}

        self.size = len(dataset)
        self.indices = self._indices_of_rank()
        self.epoch = 0

    def _infinite_indices(self) -> Iterator:
        """Infinitely yield a sequence of indices."""
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        while True:
            if self.shuffle:
                yield from torch.randperm(self.size, generator=g).tolist()
            else:
                yield from torch.arange(self.size).tolist()

    def _indices_of_rank(self) -> Iterator:
        """Slice the infinite indices by rank."""
        yield from itertools.islice(self._infinite_indices(), 0, None)

    def __iter__(self) -> Iterator:
        # once batch size is reached, yield the indices
        for idx in self.indices:
            flag = self.flag[idx]
            group_buffer = self.buffer_per_group[flag]
            group_buffer.append(idx)
            if len(group_buffer) == self.samples_per_gpu:
                for i in range(self.samples_per_gpu):
                    yield group_buffer[i]
                del group_buffer[:]

    def __len__(self) -> int:
        """Length of dataset."""
        # The length of sampler is set to the actual length of dataset, thus
        # the length of dataloader is still determined by the dataset.
        return self.size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class DistributedInfiniteSampler(Sampler):
    """Similar to `InfiniteSampler` but in distributed version.

    The length of sampler is set to the actual length of dataset, thus the
    length of dataloader is still determined by the dataset. The
    implementation logic is referred to
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/samplers/grouped_batch_sampler.py

    Args:
        dataset (Iterable): The dataset.
        num_replicas (int | None): Number of processes participating in
            distributed training. Default: None.
        rank (int | None): Rank of current process. Default: None.
        seed (int): Random seed. Default: 0.
        shuffle (bool): Whether shuffle the dataset or not. Default: True.
    """  # noqa: W605

    def __init__(self,
                 dataset: Iterable,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 seed: int = 0,
                 shuffle: bool = True) -> None:
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        self.rank = rank
        self.num_replicas = num_replicas
        self.dataset = dataset
        # In distributed sampling, different ranks should sample
        # non-overlapped data in the dataset. Therefore, this function
        # is used to make sure that each rank shuffles the data indices
        # in the same order based on the same seed. Then different ranks
        # could use different indices to select non-overlapped data from the
        # same data list.
        self.seed = sync_random_seed(seed)
        self.shuffle = shuffle
        self.size = len(dataset)
        self.indices = self._indices_of_rank()
        self.epoch = 0

    def _infinite_indices(self) -> Iterator:
        """Infinitely yield a sequence of indices."""
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        while True:
            if self.shuffle:
                indices = []
                for _ in range(self.num_replicas):
                    indices += torch.randperm(self.size, generator=g).tolist()
                yield from indices
            else:
                yield from torch.arange(self.size).tolist()

    def _indices_of_rank(self) -> Iterator:
        """Slice the infinite indices by rank."""
        yield from itertools.islice(self._infinite_indices(), self.rank, None,
                                    self.num_replicas)

    def __iter__(self) -> Iterator:
        yield from self.indices

    def __len__(self):
        """return length of dataset."""
        # The length of sampler is set to the actual length of dataset, thus
        # the length of dataloader is still determined by the dataset.
        return math.ceil(self.size / self.num_replicas)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class DistributedInfiniteGroupSampler(Sampler):
    """Similar to `InfiniteGroupSampler` but in distributed version.

    The length of sampler is set to the actual length of dataset, thus the
    length of dataloader is still determined by the dataset. The
    implementation logic is referred to
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/samplers/grouped_batch_sampler.py

    Args:
        dataset (Iterable): The dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU. Default: 1.
        num_replicas (int | None): Number of processes participating in
            distributed training. Default: None.
        rank (int | None): Rank of current process. Default: None.
        seed (int): Random seed. Default: 0.
        shuffle (bool): Whether shuffle the indices of a dummy `epoch`, it
            should be noted that `shuffle` can not guarantee that you can
            generate sequential indices because it need to ensure
            that all indices in a batch is in a group. Default: True.
    """  # noqa: W605

    def __init__(self,
                 dataset: Iterable,
                 samples_per_gpu: int = 1,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 seed: int = 0,
                 shuffle: bool = True) -> None:
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        self.rank = rank
        self.num_replicas = num_replicas
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        # In distributed sampling, different ranks should sample
        # non-overlapped data in the dataset. Therefore, this function
        # is used to make sure that each rank shuffles the data indices
        # in the same order based on the same seed. Then different ranks
        # could use different indices to select non-overlapped data from the
        # same data list.
        self.seed = sync_random_seed(seed)
        self.shuffle = shuffle

        assert hasattr(self.dataset, 'flag')
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)
        # buffer used to save indices of each group
        self.buffer_per_group = {k: [] for k in range(len(self.group_sizes))}

        self.size = len(dataset)
        self.indices = self._indices_of_rank()
        self.epoch = 0

    def _infinite_indices(self) -> Iterator:
        """Infinitely yield a sequence of indices."""
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        while True:
            if self.shuffle:
                indices = []
                for _ in range(self.num_replicas):
                    indices += torch.randperm(self.size, generator=g).tolist()
                yield from indices
            else:
                yield from torch.arange(self.size).tolist()

    def _indices_of_rank(self) -> Iterator:
        """Slice the infinite indices by rank."""
        yield from itertools.islice(self._infinite_indices(), self.rank, None,
                                    self.num_replicas)

    def __iter__(self) -> Iterator:
        # once batch size is reached, yield the indices
        for idx in self.indices:
            flag = self.flag[idx]
            group_buffer = self.buffer_per_group[flag]
            group_buffer.append(idx)
            if len(group_buffer) == self.samples_per_gpu:
                for i in range(self.samples_per_gpu):
                    yield group_buffer[i]
                del group_buffer[:]

    def __len__(self) -> int:
        """return length of dataset."""
        # The length of sampler is set to the actual length of dataset, thus
        # the length of dataloader is still determined by the dataset.
        return math.ceil(self.size / self.num_replicas)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
