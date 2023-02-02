# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

from mmcls.models.utils.augment.builder import AUGMENT
from mmcls.models.utils.augment.cutmix import BaseCutMixLayer


@AUGMENT.register_module(name='BatchCutMixTwoLabel')
class BatchCutMixLayer(BaseCutMixLayer):
    r"""CutMix layer for a batch of data.

    CutMix is a method to improve the network's generalization capability. It's
    proposed in `CutMix: Regularization Strategy to Train Strong Classifiers
    with Localizable Features <https://arxiv.org/abs/1905.04899>`

    With this method, patches are cut and pasted among training images where
    the ground truth labels are also mixed proportionally to the area of the
    patches.

    Args:
        alpha (float): Parameters for Beta distribution to generate the
            mixing ratio. It should be a positive number. More details
            can be found in :class:`BatchMixupLayer`.
        num_classes (int): The number of classes
        prob (float): The probability to execute cutmix. It should be in
            range [0, 1]. Defaults to 1.0.
        cutmix_minmax (List[float], optional): The min/max area ratio of the
            patches. If not None, the bounding-box of patches is uniform
            sampled within this ratio range, and the ``alpha`` will be ignored.
            Otherwise, the bounding-box is generated according to the
            ``alpha``. Defaults to None.
        correct_lam (bool): Whether to apply lambda correction when cutmix bbox
            clipped by image borders. Defaults to True.

    Note:
        If the ``cutmix_minmax`` is None, how to generate the bounding-box of
        patches according to the ``alpha``?

        First, generate a :math:`\lambda`, details can be found in
        :class:`BatchMixupLayer`. And then, the area ratio of the bounding-box
        is calculated by:

        .. math::
            \text{ratio} = \sqrt{1-\lambda}
    """

    def __init__(self, *args, **kwargs):
        super(BatchCutMixLayer, self).__init__(*args, **kwargs)

    def cutmix(self, img, gt_label):
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = img.size(0)
        index = torch.randperm(batch_size)

        (bby1, bby2, bbx1, bbx2), lam = self.cutmix_bbox_and_lam(img.shape, lam)
        img[:, :, bby1:bby2, bbx1:bbx2] = img[index, :, bby1:bby2, bbx1:bbx2]
        return img, lam, gt_label, gt_label[index]

    def __call__(self, img, gt_label):
        return self.cutmix(img, gt_label)
