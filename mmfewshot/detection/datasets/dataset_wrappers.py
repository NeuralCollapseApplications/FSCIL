# Copyright (c) OpenMMLab. All rights reserved.
import copy
import json
import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from mmdet.datasets.builder import DATASETS

from .base import BaseFewShotDataset


@DATASETS.register_module()
class QueryAwareDataset:
    """A wrapper of QueryAwareDataset.

    Building QueryAwareDataset requires query and support dataset.
    Every call of `__getitem__` will firstly sample a query image and its
    annotations. Then it will use the query annotations to sample a batch
    of positive and negative support images and annotations. The positive
    images share same classes with query, while the annotations of negative
    images don't have any category from query.

    Args:
        query_dataset (:obj:`BaseFewShotDataset`):
            Query dataset to be wrapped.
        support_dataset (:obj:`BaseFewShotDataset` | None):
            Support dataset to be wrapped. If support dataset is None,
            support dataset will copy from query dataset.
        num_support_ways (int): Number of classes for support in
            mini-batch, the first one always be the positive class.
        num_support_shots (int): Number of support shots for each
            class in mini-batch, the first K shots always from positive class.
        repeat_times (int): The length of repeated dataset will be `times`
            larger than the original dataset. Default: 1.
    """

    def __init__(self,
                 query_dataset: BaseFewShotDataset,
                 support_dataset: Optional[BaseFewShotDataset],
                 num_support_ways: int,
                 num_support_shots: int,
                 repeat_times: int = 1) -> None:
        self.query_dataset = query_dataset
        if support_dataset is None:
            self.support_dataset = self.query_dataset
        else:
            self.support_dataset = support_dataset
        self.num_support_ways = num_support_ways
        self.num_support_shots = num_support_shots
        self.CLASSES = self.query_dataset.CLASSES
        self.repeat_times = repeat_times
        assert self.num_support_ways <= len(
            self.CLASSES
        ), 'Please set `num_support_ways` smaller than the number of classes.'
        # build data index (idx, gt_idx) by class.
        self.data_infos_by_class = {i: [] for i in range(len(self.CLASSES))}
        # counting max number of annotations in one image for each class,
        # which will decide whether sample repeated instance or not.
        self.max_anns_num_one_image = [0 for _ in range(len(self.CLASSES))]
        # count image for each class annotation when novel class only
        # has one image, the positive support is allowed sampled from itself.
        self.num_image_by_class = [0 for _ in range(len(self.CLASSES))]

        for idx in range(len(self.support_dataset)):
            labels = self.support_dataset.get_ann_info(idx)['labels']
            class_count = [0 for _ in range(len(self.CLASSES))]
            for gt_idx, gt in enumerate(labels):
                self.data_infos_by_class[gt].append((idx, gt_idx))
                class_count[gt] += 1
            for i in range(len(self.CLASSES)):
                # number of images for each class
                if class_count[i] > 0:
                    self.num_image_by_class[i] += 1
                # max number of one class annotations in one image
                if class_count[i] > self.max_anns_num_one_image[i]:
                    self.max_anns_num_one_image[i] = class_count[i]

        for i in range(len(self.CLASSES)):
            assert len(self.data_infos_by_class[i]
                       ) > 0, f'Class {self.CLASSES[i]} has zero annotation'
            if len(
                    self.data_infos_by_class[i]
            ) <= self.num_support_shots - self.max_anns_num_one_image[i]:
                warnings.warn(
                    f'During training, instances of class {self.CLASSES[i]} '
                    f'may smaller than the number of support shots which '
                    f'causes some instance will be sampled multiple times')
            if self.num_image_by_class[i] == 1:
                warnings.warn(f'Class {self.CLASSES[i]} only have one '
                              f'image, query and support will sample '
                              f'from instance of same image')

        # Disable the group sampler, because in few shot setting,
        # one group may only has two or three images.
        if hasattr(self.query_dataset, 'flag'):
            self.flag = np.zeros(
                len(self.query_dataset) * self.repeat_times, dtype=np.uint8)

        self._ori_len = len(self.query_dataset)

    def __getitem__(self, idx: int) -> Dict:
        """Return query image and support images at the same time.

        For query aware dataset, this function would return one query image
        and num_support_ways * num_support_shots support images. The support
        images are sampled according to the selected query image. There should
        be no intersection between the classes of instances in query data and
        in support data.

        Args:
            idx (int): the index of data.

        Returns:
            dict: A dict contains query data and support data, it
            usually contains two fields.

                - query_data: A dict of single query data information.
                - support_data: A list of dict, has
                  num_support_ways * num_support_shots support images
                  and corresponding annotations.
        """
        idx %= self._ori_len
        # sample query data
        try_time = 0
        while True:
            try_time += 1
            cat_ids = self.query_dataset.get_cat_ids(idx)
            # query image have too many classes, can not find enough
            # negative support classes.
            if len(self.CLASSES) - len(cat_ids) >= self.num_support_ways - 1:
                break
            else:
                idx = self._rand_another(idx) % self._ori_len
            assert try_time < 100, \
                'Not enough negative support classes for ' \
                'query image, please try a smaller support way.'

        query_class = np.random.choice(cat_ids)
        query_gt_idx = [
            i for i in range(len(cat_ids)) if cat_ids[i] == query_class
        ]
        query_data = self.query_dataset.prepare_train_img(
            idx, 'query', query_gt_idx)
        query_data['query_class'] = [query_class]

        # sample negative support classes, which not appear in query image
        support_class = [
            i for i in range(len(self.CLASSES)) if i not in cat_ids
        ]
        support_class = np.random.choice(
            support_class,
            min(self.num_support_ways - 1, len(support_class)),
            replace=False)
        support_idxes = self.generate_support(idx, query_class, support_class)
        support_data = [
            self.support_dataset.prepare_train_img(idx, 'support', [gt_idx])
            for (idx, gt_idx) in support_idxes
        ]
        return {'query_data': query_data, 'support_data': support_data}

    def __len__(self) -> int:
        """Length after repetition."""
        return len(self.query_dataset) * self.repeat_times

    def _rand_another(self, idx: int) -> int:
        """Get another random index from the same group as the given index."""
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def generate_support(self, idx: int, query_class: int,
                         support_classes: List[int]) -> List[Tuple[int]]:
        """Generate support indices of query images.

        Args:
            idx (int): Index of query data.
            query_class (int): Query class.
            support_classes (list[int]): Classes of support data.

        Returns:
            list[tuple(int)]: A mini-batch (num_support_ways *
                num_support_shots) of support data (idx, gt_idx).
        """
        support_idxes = []
        if self.num_image_by_class[query_class] == 1:
            # only have one image, instance will sample from same image
            pos_support_idxes = self.sample_support_shots(
                idx, query_class, allow_same_image=True)
        else:
            # instance will sample from different image from query image
            pos_support_idxes = self.sample_support_shots(idx, query_class)
        support_idxes.extend(pos_support_idxes)
        for support_class in support_classes:
            neg_support_idxes = self.sample_support_shots(idx, support_class)
            support_idxes.extend(neg_support_idxes)
        return support_idxes

    def sample_support_shots(
            self,
            idx: int,
            class_id: int,
            allow_same_image: bool = False) -> List[Tuple[int]]:
        """Generate support indices according to the class id.

        Args:
            idx (int): Index of query data.
            class_id (int): Support class.
            allow_same_image (bool): Allow instance sampled from same image
                as query image. Default: False.
        Returns:
            list[tuple[int]]: Support data (num_support_shots)
                of specific class.
        """
        support_idxes = []
        num_total_shots = len(self.data_infos_by_class[class_id])

        # count number of support instance in query image
        cat_ids = self.support_dataset.get_cat_ids(idx % self._ori_len)
        num_ignore_shots = len([1 for cat_id in cat_ids if cat_id == class_id])

        # set num_sample_shots for each time of sampling
        if num_total_shots - num_ignore_shots < self.num_support_shots:
            # if not have enough support data allow repeated data
            num_sample_shots = num_total_shots
            allow_repeat = True
        else:
            # if have enough support data not allow repeated data
            num_sample_shots = self.num_support_shots
            allow_repeat = False
        while len(support_idxes) < self.num_support_shots:
            selected_gt_idxes = np.random.choice(
                num_total_shots, num_sample_shots, replace=False)

            selected_gts = [
                self.data_infos_by_class[class_id][selected_gt_idx]
                for selected_gt_idx in selected_gt_idxes
            ]
            for selected_gt in selected_gts:
                # filter out query annotations
                if selected_gt[0] == idx:
                    if not allow_same_image:
                        continue
                if allow_repeat:
                    support_idxes.append(selected_gt)
                elif selected_gt not in support_idxes:
                    support_idxes.append(selected_gt)
                if len(support_idxes) == self.num_support_shots:
                    break
            # update the number of data for next time sample
            num_sample_shots = min(self.num_support_shots - len(support_idxes),
                                   num_sample_shots)
        return support_idxes

    def save_data_infos(self, output_path: str) -> None:
        """Save data_infos into json."""
        self.query_dataset.save_data_infos(output_path)
        # for query aware datasets support and query set use same data
        paths = output_path.split('.')
        self.support_dataset.save_data_infos(
            '.'.join(paths[:-1] + ['support_shot', paths[-1]]))

    def get_support_data_infos(self) -> List[Dict]:
        """Return data_infos of support dataset."""
        return copy.deepcopy(self.support_dataset.data_infos)


@DATASETS.register_module()
class NWayKShotDataset:
    """A dataset wrapper of NWayKShotDataset.

    Building NWayKShotDataset requires query and support dataset, the behavior
    of NWayKShotDataset is determined by `mode`. When dataset in 'query' mode,
    dataset will return regular image and annotations. While dataset in
    'support' mode, dataset will build batch indices firstly and each batch
    indices contain (num_support_ways * num_support_shots) samples. In other
    words, for support mode every call of `__getitem__` will return a batch
    of samples, therefore the outside dataloader should set batch_size to 1.
    The default `mode` of NWayKShotDataset is 'query' and by using convert
    function `convert_query_to_support` the `mode` will be converted into
    'support'.

    Args:
        query_dataset (:obj:`BaseFewShotDataset`):
            Query dataset to be wrapped.
        support_dataset (:obj:`BaseFewShotDataset` | None):
            Support dataset to be wrapped. If support dataset is None,
            support dataset will copy from query dataset.
        num_support_ways (int): Number of classes for support in
            mini-batch.
        num_support_shots (int): Number of support shot for each
            class in mini-batch.
        one_support_shot_per_image (bool): If True only one annotation will be
            sampled from each image. Default: False.
        num_used_support_shots (int | None): The total number of support
            shots sampled and used for each class during training. If set to
            None, all shots in dataset will be used as support shot.
            Default: 200.
        shuffle_support (bool): If allow generate new batch indices for
            each epoch. Default: False.
        repeat_times (int): The length of repeated dataset will be `times`
            larger than the original dataset. Default: 1.
    """

    def __init__(self,
                 query_dataset: BaseFewShotDataset,
                 support_dataset: Optional[BaseFewShotDataset],
                 num_support_ways: int,
                 num_support_shots: int,
                 one_support_shot_per_image: bool = False,
                 num_used_support_shots: int = 200,
                 repeat_times: int = 1) -> None:
        self.query_dataset = query_dataset
        if support_dataset is None:
            self.support_dataset = self.query_dataset
        else:
            self.support_dataset = support_dataset
        self.CLASSES = self.query_dataset.CLASSES
        # The mode determinate the behavior of fetching data,
        # the default mode is 'query'. To convert the dataset
        # into 'support' dataset, simply call the function
        # convert_query_to_support().
        self._mode = 'query'
        self.num_support_ways = num_support_ways
        self.one_support_shot_per_image = one_support_shot_per_image
        self.num_used_support_shots = num_used_support_shots
        assert num_support_ways <= len(
            self.CLASSES
        ), 'support way can not larger than the number of classes'
        self.num_support_shots = num_support_shots
        self.batch_indices = []
        self.data_infos_by_class = {i: [] for i in range(len(self.CLASSES))}
        self.prepare_support_shots()
        self.repeat_times = repeat_times
        # Disable the group sampler, because in few shot setting,
        # one group may only has two or three images.
        if hasattr(query_dataset, 'flag'):
            self.flag = np.zeros(
                len(self.query_dataset) * self.repeat_times, dtype=np.uint8)

        self._ori_len = len(self.query_dataset)

    def __getitem__(self, idx: int) -> Union[Dict, List[Dict]]:
        if self._mode == 'query':
            idx %= self._ori_len
            # loads one data in query pipeline
            return self.query_dataset.prepare_train_img(idx, 'query')
        elif self._mode == 'support':
            # loads one batch of data in support pipeline
            b_idx = self.batch_indices[idx]
            batch_data = [
                self.support_dataset.prepare_train_img(idx, 'support',
                                                       [gt_idx])
                for (idx, gt_idx) in b_idx
            ]
            return batch_data
        else:
            raise ValueError('not valid data type')

    def __len__(self) -> int:
        """Length of dataset."""
        if self._mode == 'query':
            return len(self.query_dataset) * self.repeat_times
        elif self._mode == 'support':
            return len(self.batch_indices)
        else:
            raise ValueError(f'{self._mode}not a valid mode')

    def prepare_support_shots(self) -> None:
        # create lookup table for annotations in same class.
        for idx in range(len(self.support_dataset)):
            labels = self.support_dataset.get_ann_info(idx)['labels']
            for gt_idx, gt in enumerate(labels):
                # When the number of support shots reaches
                # `num_used_support_shots`, the class will be skipped
                if len(self.data_infos_by_class[gt]) < \
                        self.num_used_support_shots:
                    self.data_infos_by_class[gt].append((idx, gt_idx))
                    # When `one_support_shot_per_image` is true, only one
                    # annotation will be sampled for each image.
                    if self.one_support_shot_per_image:
                        break
        # make sure all class index lists have enough
        # instances (length > num_support_shots)
        for i in range(len(self.CLASSES)):
            num_gts = len(self.data_infos_by_class[i])
            if num_gts < self.num_support_shots:
                self.data_infos_by_class[i] = self.data_infos_by_class[i] * (
                    self.num_support_shots // num_gts + 1)

    def convert_query_to_support(self, support_dataset_len: int) -> None:
        """Convert query dataset to support dataset.

        Args:
            support_dataset_len (int): Length of pre sample batch indices.
        """
        self.batch_indices = \
            self.generate_support_batch_indices(support_dataset_len)
        self._mode = 'support'
        if hasattr(self, 'flag'):
            self.flag = np.zeros(support_dataset_len, dtype=np.uint8)

    def generate_support_batch_indices(
            self, dataset_len: int) -> List[List[Tuple[int]]]:
        """Generate batch indices from support dataset.

        Batch indices is in the shape of [length of datasets * [support way *
        support shots]]. And the `dataset_len` will be the length of support
        dataset.

        Args:
            dataset_len (int): Length of batch indices.

        Returns:
            list[list[(data_idx, gt_idx)]]: Pre-sample batch indices.
        """
        total_indices = []
        for _ in range(dataset_len):
            batch_indices = []
            selected_classes = np.random.choice(
                len(self.CLASSES), self.num_support_ways, replace=False)
            for cls in selected_classes:
                num_gts = len(self.data_infos_by_class[cls])
                selected_gts_idx = np.random.choice(
                    num_gts, self.num_support_shots, replace=False)
                selected_gts = [
                    self.data_infos_by_class[cls][gt_idx]
                    for gt_idx in selected_gts_idx
                ]
                batch_indices.extend(selected_gts)
            total_indices.append(batch_indices)
        return total_indices

    def save_data_infos(self, output_path: str) -> None:
        """Save data infos of query and support data."""
        self.query_dataset.save_data_infos(output_path)
        paths = output_path.split('.')
        self.save_support_data_infos('.'.join(paths[:-1] +
                                              ['support_shot', paths[-1]]))

    def save_support_data_infos(self, support_output_path: str) -> None:
        """Save support data infos."""
        support_data_infos = self.get_support_data_infos()
        meta_info = [{
            'CLASSES': self.CLASSES,
            'img_prefix': self.support_dataset.img_prefix
        }]
        from .utils import NumpyEncoder
        with open(support_output_path, 'w', encoding='utf-8') as f:
            json.dump(
                meta_info + support_data_infos,
                f,
                ensure_ascii=False,
                indent=4,
                cls=NumpyEncoder)

    def get_support_data_infos(self) -> List[Dict]:
        """Get support data infos from batch indices."""
        return copy.deepcopy([
            self._get_shot_data_info(idx, gt_idx)
            for class_name in self.data_infos_by_class.keys()
            for (idx, gt_idx) in self.data_infos_by_class[class_name]
        ])

    def _get_shot_data_info(self, idx: int, gt_idx: [int]) -> Dict:
        """Get data info by idx and gt idx."""
        data_info = copy.deepcopy(self.support_dataset.data_infos[idx])
        data_info['ann']['labels'] = \
            data_info['ann']['labels'][gt_idx:gt_idx + 1]
        data_info['ann']['bboxes'] = \
            data_info['ann']['bboxes'][gt_idx:gt_idx + 1]
        return data_info


@DATASETS.register_module()
class TwoBranchDataset:
    """A dataset wrapper of TwoBranchDataset.

    Wrapping main_dataset and auxiliary_dataset to a single dataset and thus
    building TwoBranchDataset requires two dataset. The behavior of
    TwoBranchDataset is determined by `mode`. Dataset will return images
    and annotations according to `mode`, e.g. fetching data from
    main_dataset if `mode` is 'main'. The default `mode` is 'main' and
    by using convert function `convert_main_to_auxiliary` the `mode`
    will be converted into 'auxiliary'.

    Args:
        main_dataset (:obj:`BaseFewShotDataset`):
            Main dataset to be wrapped.
        auxiliary_dataset (:obj:`BaseFewShotDataset` | None):
            Auxiliary dataset to be wrapped. If auxiliary dataset is None,
            auxiliary dataset will copy from main dataset.
        reweight_dataset (bool): Whether to change the sampling weights
            of VOC07 and VOC12 . Default: False.
    """

    def __init__(self,
                 main_dataset: BaseFewShotDataset = None,
                 auxiliary_dataset: Optional[BaseFewShotDataset] = None,
                 reweight_dataset: bool = False) -> None:
        assert main_dataset and auxiliary_dataset
        self._mode = 'main'
        self.main_dataset = main_dataset
        self.auxiliary_dataset = auxiliary_dataset
        self.CLASSES = self.main_dataset.CLASSES
        if reweight_dataset:
            # Reweight the VOC dataset to be consistent with the original
            # implementation of MPSR. For more details, please refer to
            # https://github.com/jiaxi-wu/MPSR/blob/master/maskrcnn_benchmark/data/datasets/voc.py#L137
            self.main_idx_map = self.reweight_dataset(
                self.main_dataset,
                ['VOC2007', 'VOC2012'],
            )
            self.auxiliary_idx_map = self.reweight_dataset(
                self.auxiliary_dataset, ['VOC'])
        else:
            self.main_idx_map = list(range(len(self.main_dataset)))
            self.auxiliary_idx_map = list(range(len(self.auxiliary_dataset)))
        self._main_len = len(self.main_idx_map)
        self._auxiliary_len = len(self.auxiliary_idx_map)
        self._set_group_flag()

    def __getitem__(self, idx: int) -> Dict:
        if self._mode == 'main':
            idx %= self._main_len
            idx = self.main_idx_map[idx]
            return self.main_dataset.prepare_train_img(idx, 'main')
        elif self._mode == 'auxiliary':
            idx %= self._auxiliary_len
            idx = self.auxiliary_idx_map[idx]
            return self.auxiliary_dataset.prepare_train_img(idx, 'auxiliary')
        else:
            raise ValueError('not valid data type')

    def __len__(self) -> int:
        """Length of dataset."""
        if self._mode == 'main':
            return self._main_len
        elif self._mode == 'auxiliary':
            return self._auxiliary_len
        else:
            raise ValueError('not valid data type')

    def convert_main_to_auxiliary(self) -> None:
        """Convert main dataset to auxiliary dataset."""
        self._mode = 'auxiliary'
        self._set_group_flag()

    def save_data_infos(self, output_path: str) -> None:
        """Save data infos of main and auxiliary data."""
        self.main_dataset.save_data_infos(output_path)
        paths = output_path.split('.')
        self.auxiliary_dataset.save_data_infos(
            '.'.join(paths[:-1] + ['auxiliary', paths[-1]]))

    def _set_group_flag(self) -> None:
        # disable the group sampler, because in few shot setting,
        # one group may only has two or three images.
        self.flag = np.zeros(len(self), dtype=np.uint8)

    @staticmethod
    def reweight_dataset(dataset: BaseFewShotDataset,
                         group_prefix: Sequence[str],
                         repeat_length: int = 100) -> List:
        """Reweight the dataset."""

        groups = [[] for _ in range(len(group_prefix))]
        for i in range(len(dataset)):
            filename = dataset.data_infos[i]['filename']
            for j, prefix in enumerate(group_prefix):
                if prefix in filename:
                    groups[j].append(i)
                    break
                assert j < len(group_prefix) - 1

        # Reweight the dataset to be consistent with the original
        # implementation of MPSR. For more details, please refer to
        # https://github.com/jiaxi-wu/MPSR/blob/master/maskrcnn_benchmark/data/datasets/voc.py#L137
        reweight_idx_map = []
        for g in groups:
            if len(g) < 50:
                reweight_idx_map += g * (int(repeat_length / len(g)) + 1)
            else:
                reweight_idx_map += g
        return reweight_idx_map
