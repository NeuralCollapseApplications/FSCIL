# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

from mmcls.models.utils.augment.builder import AUGMENT
from mmcls.models.utils.augment.mixup import BaseMixupLayer


@AUGMENT.register_module(name='BatchMixupTwoLabel')
class BatchMixupLayer(BaseMixupLayer):
    r"""Mixup layer for a batch of data.

    Mixup is a method to reduces the memorization of corrupt labels and
    increases the robustness to adversarial examples. It's
    proposed in `mixup: Beyond Empirical Risk Minimization
    <https://arxiv.org/abs/1710.09412>`

    This method simply linearly mix pairs of data and their labels.

    Args:
        alpha (float): Parameters for Beta distribution to generate the
            mixing ratio. It should be a positive number. More details
            are in the note.
        num_classes (int): The number of classes.
        prob (float): The probability to execute mixup. It should be in
            range [0, 1]. Default sto 1.0.

    Note:
        The :math:`\alpha` (``alpha``) determines a random distribution
        :math:`Beta(\alpha, \alpha)`. For each batch of data, we sample
        a mixing ratio (marked as :math:`\lambda`, ``lam``) from the random
        distribution.
    """

    def __init__(self, *args, **kwargs):
        super(BatchMixupLayer, self).__init__(*args, **kwargs)

    def mixup(self, img, gt_label):
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = img.size(0)
        index = torch.randperm(batch_size)

        mixed_img = lam * img + (1 - lam) * img[index, :]
        return mixed_img, lam,  gt_label, gt_label[index]

    def __call__(self, img, gt_label):
        return self.mixup(img, gt_label)
