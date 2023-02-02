# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, Optional

import torch
import torch.nn as nn
from mmcv.runner import force_fp32
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import accuracy
from mmdet.models.roi_heads import BBoxHead
from torch import Tensor


@HEADS.register_module()
class MetaBBoxHead(BBoxHead):
    """BBoxHead with meta classification for metarcnn and fsdetview.

    Args:
        num_meta_classes (int): Number of classes for meta classification.
        meta_cls_in_channels (int): Number of support feature channels.
        with_meta_cls_loss (bool): Use meta classification loss.
            Default: True.
        meta_cls_loss_weight (float | None): The loss weight of `loss_meta`.
            Default: None.
        loss_meta (dict): Config for meta classification loss.
    """

    def __init__(self,
                 num_meta_classes: int,
                 meta_cls_in_channels: int = 2048,
                 with_meta_cls_loss: bool = True,
                 meta_cls_loss_weight: Optional[float] = None,
                 loss_meta: Dict = dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.with_meta_cls_loss = with_meta_cls_loss
        if with_meta_cls_loss:
            self.fc_meta = nn.Linear(meta_cls_in_channels, num_meta_classes)
            self.meta_cls_loss_weight = meta_cls_loss_weight
            self.loss_meta_cls = build_loss(copy.deepcopy(loss_meta))

    def forward_meta_cls(self, support_feat: Tensor) -> Tensor:
        """Forward function for meta classification.

        Args:
            support_feat (Tensor): Shape of (N, C, H, W).

        Returns:
            Tensor: Box scores with shape of (N, num_meta_classes, H, W).
        """
        meta_cls_score = self.fc_meta(support_feat)
        return meta_cls_score

    @force_fp32(apply_to='meta_cls_score')
    def loss_meta(self,
                  meta_cls_score: Tensor,
                  meta_cls_labels: Tensor,
                  meta_cls_label_weights: Tensor,
                  reduction_override: Optional[str] = None) -> Dict:
        """Meta classification loss.

        Args:
            meta_cls_score (Tensor): Predicted meta classification scores
                 with shape (N, num_meta_classes).
            meta_cls_labels (Tensor): Corresponding class indices with
                shape (N).
            meta_cls_label_weights (Tensor): Meta classification loss weight
                of each sample with shape (N).
            reduction_override (str | None): The reduction method used to
                override the original reduction method of the loss. Options
                are "none", "mean" and "sum". Default: None.

        Returns:
            Dict: The calculated loss.
        """
        losses = dict()
        if self.meta_cls_loss_weight is None:
            loss_weight = 1. / max(
                torch.sum(meta_cls_label_weights > 0).float().item(), 1.)
        else:
            loss_weight = self.meta_cls_loss_weight
        if meta_cls_score.numel() > 0:
            loss_meta_cls_ = self.loss_meta_cls(
                meta_cls_score,
                meta_cls_labels,
                meta_cls_label_weights,
                reduction_override=reduction_override)
            losses['loss_meta_cls'] = loss_meta_cls_ * loss_weight
            losses['meta_acc'] = accuracy(meta_cls_score, meta_cls_labels)
        return losses
