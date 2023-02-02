import os.path as osp
import pickle
from collections import defaultdict
from typing import List, Dict, Mapping, Optional, Tuple

from mmcv.runner import get_dist_info

from mmcls.datasets.utils import download_and_extract_archive, check_integrity
from typing_extensions import Literal

import copy
import numpy as np
from mmcls.datasets.builder import DATASETS
from mmcls.datasets.pipelines import Compose

from torch.utils.data import Dataset
import torch.distributed as dist

CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver',
    'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly',
    'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach',
    'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox',
    'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard',
    'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid',
    'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',

    'plain', 'plate', 'poppy', 'porcupine', 'possum',
    'rabbit', 'raccoon', 'ray', 'road', 'rocket',
    'rose', 'sea', 'seal', 'shark', 'shrew',
    'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor',
    'train', 'trout', 'tulip', 'turtle', 'wardrobe',
    'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

# Please refer to https://github.com/icoz69/CEC-CVPR2021/tree/dc26237/data/index_list/cifar100
FSCIL_SAMPLES = {
    'plain': [29774, 33344, 4815, 6772, 48317],
    'plate': [29918, 33262, 5138, 7342, 47874],
    'poppy': [28864, 32471, 4316, 6436, 47498],
    'porcupine': [29802, 33159, 3730, 5093, 47740],
    'possum': [30548, 34549, 2845, 4996, 47866],

    'rabbit': [28855, 32834, 4603, 6914, 48126],
    'raccoon': [29932, 33300, 3860, 5424, 47055],
    'ray': [29434, 32604, 4609, 6380, 47844],
    'road': [30456, 34217, 4361, 6550, 46896],
    'rocket': [29664, 32857, 4923, 7502, 47270],

    'rose': [31267, 34427, 4799, 6611, 47404],
    'sea': [28509, 31687, 3477, 5563, 48003],
    'seal': [29545, 33412, 5114, 6808, 47692],
    'shark': [29209, 33265, 4131, 6401, 48102],
    'shrew': [31290, 34432, 6060, 8451, 48279],

    'skunk': [32337, 35646, 6022, 9048, 48584],
    'skyscraper': [30768, 34394, 5091, 6510, 48023],
    'snail': [30310, 33230, 5098, 6671, 48349],
    'snake': [29690, 33490, 4260, 5916, 47371],
    'spider': [31173, 34943, 4517, 6494, 47689],

    'squirrel': [30281, 33894, 3768, 6113, 48095],
    'streetcar': [28913, 32821, 6172, 8276, 48004],
    'sunflower': [31249, 34088, 5257, 6961, 47534],
    'sweet_pepper': [30404, 34101, 4985, 6899, 48115],
    'table': [31823, 35148, 3922, 6548, 48127],

    'tank': [30815, 34450, 3481, 5089, 47913],
    'telephone': [31683, 34591, 5251, 7608, 47984],
    'television': [29837, 33823, 4615, 6448, 47752],
    'tiger': [31222, 34079, 5686, 7919, 48675],
    'tractor': [28567, 32964, 5009, 6201, 47039],

    'train': [29355, 33909, 3982, 5389, 47166],
    'trout': [31058, 35180, 5177, 6890, 48032],
    'tulip': [31176, 35098, 5235, 7861, 47830],
    'turtle': [30874, 34639, 5266, 7489, 47323],
    'wardrobe': [29960, 34050, 4988, 7434, 48208],

    'whale': [30463, 34580, 5230, 6813, 48605],
    'willow_tree': [31702, 35249, 5854, 7765, 48444],
    'wolf': [30380, 34028, 5211, 7433, 47988],
    'woman': [31348, 34021, 4929, 7033, 47904],
    'worm': [30627, 33728, 4895, 6299, 47507],
}


@DATASETS.register_module()
class CIFAR100FSCILDataset(Dataset):
    """CIRFAR100 dataset for few shot class-incremental classification.
    few_cls is None when performing usual training, is tuple for few-shot training
    """

    # Copy and paste from torchvision
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }

    def __init__(
            self,
            data_prefix: str,
            pipeline: List[Dict],
            num_cls: int = 100,
            subset: Literal['train', 'test'] = 'train',
            few_cls: Optional[Tuple] = None,
            test_mode: bool = False,
    ):
        rank, world_size = get_dist_info()

        self.data_prefix = data_prefix
        assert isinstance(pipeline, list), 'pipeline is type of list'
        self.pipeline = Compose(pipeline)

        if rank == 0 and not self._check_integrity():
            download_and_extract_archive(
                self.url,
                self.data_prefix,
                filename=self.filename,
                md5=self.tgz_md5)

        if world_size > 1:
            dist.barrier()
            assert self._check_integrity(), \
                'Shared storage seems unavailable. ' \
                f'Please download the dataset manually through {self.url}.'

        self.subset = subset
        if self.subset == 'train':
            downloaded_list = self.train_list
        elif self.subset == 'test':
            downloaded_list = self.test_list
        else:
            raise NotImplementedError

        if few_cls is not None:
            assert self.subset == 'train'
            self.CLASSES = [CLASSES[_] for _ in few_cls]
            self.few_mod = True
        else:
            self.CLASSES = self.get_classes(num_cls)
            self.few_mod = False

        self.data_infos = self.load_annotations(downloaded_list)

    @staticmethod
    def get_classes(num_cls):
        return CLASSES[:num_cls]

    @property
    def class_to_idx(self) -> Mapping:
        """Map mapping class name to class index.

        Returns:
            dict: mapping from class name to class index.
        """

        return {_class: i for i, _class in enumerate(CLASSES)}

    def load_annotations(self, downloaded_list) -> List:
        """Load annotation according to the classes subset."""
        imgs = []
        gt_labels = []

        # load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = osp.join(self.data_prefix, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                imgs.append(entry['data'])
                if 'labels' in entry:
                    gt_labels.extend(entry['labels'])
                else:
                    gt_labels.extend(entry['fine_labels'])

        imgs = np.vstack(imgs).reshape(-1, 3, 32, 32)
        imgs = imgs.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

        data_infos = []

        cls_cnt = defaultdict(lambda: 0)
        if self.few_mod:
            for cls in self.CLASSES:
                for idx in FSCIL_SAMPLES[cls]:
                    assert CLASSES[gt_labels[idx]] == cls
                    info = {
                        'img': imgs[idx],
                        'gt_label': gt_labels[idx],
                        'cls_id': self.class_to_idx[cls],
                        'img_id': cls_cnt[cls]
                    }
                    cls_cnt[cls] += 1
                    data_infos.append(info)
        else:
            for img, _gt_label in zip(imgs, gt_labels):
                if CLASSES[_gt_label] in self.CLASSES:
                    gt_label = np.array(_gt_label, dtype=np.int64)
                    info = {'img': img, 'gt_label': gt_label, 'cls_id': _gt_label, 'img_id': cls_cnt[_gt_label]}
                    cls_cnt[_gt_label] += 1
                    data_infos.append(info)
        return data_infos

    def __len__(self) -> int:
        """Return length of the dataset."""
        return len(self.data_infos)

    def __getitem__(self, idx: int) -> Dict:
        return self.prepare_data(idx)

    def prepare_data(self, idx: int) -> Dict:
        results = copy.deepcopy(self.data_infos[idx])
        return self.pipeline(results)

    # from mmcls, thx
    def _load_meta(self):
        path = osp.join(self.data_prefix, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError(
                'Dataset metadata file not found or corrupted.' +
                ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            for idx, name in enumerate(data[self.meta['key']]):
                assert CLASSES[idx] == name

    # from mmcls, thx
    def _check_integrity(self):
        root = self.data_prefix
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = osp.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

