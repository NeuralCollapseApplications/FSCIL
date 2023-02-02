# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import sys

import torch.distributed as dist
from mmcv.runner import DistEvalHook as BaseDistEvalHook
from mmcv.runner import EvalHook as BaseEvalHook
from mmcv.runner import Runner
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import DataLoader


class QuerySupportEvalHook(BaseEvalHook):
    """Evaluation hook for query support data pipeline.

    This hook will first traverse `model_init_dataloader` to extract support
    features for model initialization and then evaluate the data from
    `val_dataloader`.

    Args:
        model_init_dataloader (DataLoader): A PyTorch dataloader of
            `model_init` dataset.
        val_dataloader (DataLoader): A PyTorch dataloader of dataset to be
            evaluated.
        **eval_kwargs: Evaluation arguments fed into the evaluate function of
            the dataset.
    """

    def __init__(self, model_init_dataloader: DataLoader,
                 val_dataloader: DataLoader, **eval_kwargs) -> None:
        super().__init__(val_dataloader, **eval_kwargs)
        self.model_init_dataloader = model_init_dataloader

    def _do_evaluate(self, runner: Runner) -> None:
        """perform evaluation and save checkpoint."""
        if not self._should_evaluate(runner):
            return
        # In meta-learning based detectors, model forward use `mode` to
        # identified the 'train', 'val' and 'model_init' stages instead
        # of `return_loss` in mmdet. Thus, `single_gpu_test` should be
        # imported from mmfewshot.
        from mmfewshot.detection.apis import (single_gpu_model_init,
                                              single_gpu_test)

        # `single_gpu_model_init` extracts features from
        # `model_init_dataloader` for model initialization with single gpu.
        single_gpu_model_init(runner.model, self.model_init_dataloader)
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = self.evaluate(runner, results)
        if self.save_best:
            self._save_ckpt(runner, key_score)


class QuerySupportDistEvalHook(BaseDistEvalHook):
    """Distributed evaluation hook for query support data pipeline.

    This hook will first traverse `model_init_dataloader` to extract support
    features for model initialization and then evaluate the data from
    `val_dataloader`.

    Noted that `model_init_dataloader` should NOT use distributed sampler
    to make all the models on different gpus get same data results
    in same initialized models.

    Args:
        model_init_dataloader (DataLoader): A PyTorch dataloader of
            `model_init` dataset.
        val_dataloader (DataLoader): A PyTorch dataloader of dataset to be
            evaluated.
        **eval_kwargs: Evaluation arguments fed into the evaluate function of
            the dataset.
    """

    def __init__(self, model_init_dataloader: DataLoader,
                 val_dataloader: DataLoader, **eval_kwargs) -> None:
        super().__init__(val_dataloader, **eval_kwargs)
        self.model_init_dataloader = model_init_dataloader

    def _do_evaluate(self, runner: Runner) -> None:
        """perform evaluation and save checkpoint."""

        if not self._should_evaluate(runner):
            return
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        # In meta-learning based detectors, model forward use `mode` to
        # identified the 'train', 'val' and 'model_init' stages instead
        # of `return_loss` in mmdet. Thus, `multi_gpu_test` should be
        # imported from mmfewshot.
        from mmfewshot.detection.apis import (multi_gpu_model_init,
                                              multi_gpu_test)

        # Noted that `model_init_dataloader` should NOT use distributed
        # sampler to make all the models on different gpus get same data
        # results in the same initialized models.
        multi_gpu_model_init(runner.model, self.model_init_dataloader)

        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            sys.stdout.write('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results)

            if self.save_best:
                self._save_ckpt(runner, key_score)
