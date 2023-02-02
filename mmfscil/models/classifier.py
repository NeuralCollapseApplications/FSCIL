# Copyright (c) OpenMMLab. All rights reserved.
import torch
import numpy as np

from mmcls.models.builder import CLASSIFIERS, build_backbone, build_head, build_neck
from mmcls.models.heads import MultiLabelClsHead
from mmcls.models.utils.augment import Augments
from mmcls.models.classifiers.base import BaseClassifier


@CLASSIFIERS.register_module()
class ImageClassifierCIL(BaseClassifier):

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 train_cfg=None,
                 init_cfg=None,
                 mixup: float = 0.,
                 mixup_prob: float = 0.,
                 ):
        super().__init__(init_cfg)

        if pretrained is not None:
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if head is not None:
            self.head = build_head(head)

        self.augments = None
        if train_cfg is not None:
            augments_cfg = train_cfg.get('augments', None)
            if augments_cfg is not None:
                self.augments = Augments(augments_cfg)

        self.mixup = mixup
        self.mixup_prob = mixup_prob

    def extract_feat(self, img, stage='neck'):
        """Directly extract features from the specified stage.

        Args:
            img (Tensor): The input images. The shape of it should be
                ``(num_samples, num_channels, *img_shape)``.
            stage (str): Which stage to output the feature. Choose from
                "backbone", "neck" and "pre_logits". Defaults to "neck".

        Returns:
            tuple | Tensor: The output of specified stage.
                The output depends on detailed implementation. In general, the
                output of backbone and neck is a tuple and the output of
                pre_logits is a tensor.

        Examples:
            1. Backbone output

            >>> import torch
            >>> from mmcv import Config
            >>> from mmcls.models import build_classifier
            >>>
            >>> cfg = Config.fromfile('configs/resnet/resnet18_8xb32_in1k.py').model
            >>> cfg.backbone.out_indices = (0, 1, 2, 3)  # Output multi-scale feature maps
            >>> model = build_classifier(cfg)
            >>> outs = model.extract_feat(torch.rand(1, 3, 224, 224), stage='backbone')
            >>> for out in outs:
            ...     print(out.shape)
            torch.Size([1, 64, 56, 56])
            torch.Size([1, 128, 28, 28])
            torch.Size([1, 256, 14, 14])
            torch.Size([1, 512, 7, 7])

            2. Neck output

            >>> import torch
            >>> from mmcv import Config
            >>> from mmcls.models import build_classifier
            >>>
            >>> cfg = Config.fromfile('configs/resnet/resnet18_8xb32_in1k.py').model
            >>> cfg.backbone.out_indices = (0, 1, 2, 3)  # Output multi-scale feature maps
            >>> model = build_classifier(cfg)
            >>>
            >>> outs = model.extract_feat(torch.rand(1, 3, 224, 224), stage='neck')
            >>> for out in outs:
            ...     print(out.shape)
            torch.Size([1, 64])
            torch.Size([1, 128])
            torch.Size([1, 256])
            torch.Size([1, 512])

            3. Pre-logits output (without the final linear classifier head)

            >>> import torch
            >>> from mmcv import Config
            >>> from mmcls.models import build_classifier
            >>>
            >>> cfg = Config.fromfile('configs/vision_transformer/vit-base-p16_pt-64xb64_in1k-224.py').model
            >>> model = build_classifier(cfg)
            >>>
            >>> out = model.extract_feat(torch.rand(1, 3, 224, 224), stage='pre_logits')
            >>> print(out.shape)  # The hidden dims in head is 3072
            torch.Size([1, 3072])
        """  # noqa: E501
        assert stage in ['backbone', 'neck', 'pre_logits'], \
            (f'Invalid output stage "{stage}", please choose from "backbone", '
             '"neck" and "pre_logits"')

        if self.backbone is not None:
            x = self.backbone(img)
        else:
            x = img

        if stage == 'backbone':
            if isinstance(x, tuple):
                x = x[-1]
            return x

        if self.with_neck:
            x = self.neck(x)
        if stage == 'neck':
            return x

        if self.with_head and hasattr(self.head, 'pre_logits'):
            x = self.head.pre_logits(x)
        return x

    def forward_train(self, img, gt_label, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            gt_label (Tensor): It should be of shape (N, 1) encoding the
                ground-truth label of input images for single label task. It
                shoulf be of shape (N, C) encoding the ground-truth label
                of input images for multi-labels task.
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if self.augments is not None:
            img, lam, gt_label, gt_label_aux = self.augments(img, gt_label)
        else:
            gt_label_aux = None
            lam = None

        x = self.extract_feat(img)

        losses = dict()
        if self.mixup == 0.:
            loss = self.head.forward_train(x, gt_label)
            losses.update(loss)

        # make sure not mixup feat
        if gt_label_aux is not None:
            assert self.mixup == 0.
            loss = self.head.forward_train(x, gt_label_aux)
            losses['loss_main'] = losses['loss'] * lam
            losses['loss_aux'] = loss['loss'] * (1 - lam)
            del losses['loss']

        # mixup feat when mixup > 0, this cannot be with augment mixup
        if self.mixup > 0. and self.mixup_prob > 0. and np.random.random() > (1 - self.mixup_prob):
            x, gt_a, gt_b, lam = self.mixup_feat(x, gt_label, alpha=self.mixup)
            loss1 = self.head.forward_train(x, gt_a)
            loss2 = self.head.forward_train(x, gt_b)
            losses['loss'] = lam * loss1['loss'] + (1-lam) * loss2['loss']
            losses['loss_1'] = loss1['loss']
            losses['loss_2'] = loss2['loss']
            losses['accuracy'] = loss1['accuracy']
        else:
            loss = self.head.forward_train(x, gt_label)
            losses.update(loss)

        return losses

    def simple_test(self, img, gt_label, return_backbone=False, return_feat=False, return_acc=False,
                    img_metas=None, **kwargs):
        """Test without augmentation."""
        if return_backbone:
            x = self.extract_feat(img, stage='backbone')
            return x
        x = self.extract_feat(img)

        if return_feat:
            assert not return_acc
            return x

        if isinstance(self.head, MultiLabelClsHead):
            assert 'softmax' not in kwargs, (
                'Please use `sigmoid` instead of `softmax` '
                'in multi-label tasks.')
        res = self.head.simple_test(x, post_process=not return_acc, **kwargs)
        if return_acc:
            res = res.argmax(dim=-1)
            return torch.eq(res, gt_label).to(dtype=torch.float32).cpu().numpy().tolist()
        return res

    @staticmethod
    def mixup_feat(feat, gt_labels, alpha=1.0):
        if alpha > 0:
            lam = alpha
        else:
            lam = 0.5

        batch_size = feat.size()[0]
        index = torch.randperm(batch_size).to(device=feat.device)

        mixed_feat = lam * feat + (1 - lam) * feat[index, :]
        gt_a, gt_b = gt_labels, gt_labels[index]

        return mixed_feat, gt_a, gt_b, lam
