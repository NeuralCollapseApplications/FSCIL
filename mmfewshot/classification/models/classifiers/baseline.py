# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

from mmcls.models.builder import CLASSIFIERS

from .base_finetune import BaseFinetuneClassifier


@CLASSIFIERS.register_module()
class Baseline(BaseFinetuneClassifier):
    """Implementation of `Baseline <https://arxiv.org/abs/1904.04232>`_.

    Args:
        head (dict): Config of classification head for training.
        meta_test_head (dict): Config of classification head for meta testing.
            the `meta_test_head` only will be built and run in meta testing.
    """

    def __init__(self,
                 head: Dict = dict(
                     type='LinearHead', num_classes=100, in_channels=1600),
                 meta_test_head: Dict = dict(
                     type='LinearHead', num_classes=5, in_channels=1600),
                 *args,
                 **kwargs) -> None:
        super().__init__(
            head=head, meta_test_head=meta_test_head, *args, **kwargs)
