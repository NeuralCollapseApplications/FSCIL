# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import shutil

import mmcv
import numpy as np
import platform
import torch.distributed as dist
from mmcv.runner import DistEvalHook as BaseDistEvalHook
from mmcv.runner import EvalHook as BaseEvalHook
from torch.nn.modules.batchnorm import _BatchNorm


class EvalHook(BaseEvalHook):
    """Non-Distributed evaluation hook.
    """

    def __init__(self, dataloader, **kwargs):
        super(EvalHook, self).__init__(dataloader, **kwargs)

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        results = self.test_fn(runner.model, self.dataloader)
        if runner.rank == 0:
            runner.logger.info("Eval Hook : There are {} samples in total.".format(len(results)))
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = {"acc": np.array(results).mean() * 100}

        for name, val in key_score.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True
        if self.save_best:
            if self.key_indicator == 'auto':
                # infer from eval_results
                self._init_rule(self.rule, list(key_score.keys())[0])
            self._save_ckpt(runner, key_score[self.key_indicator])

            dst_file = osp.join(self.out_dir, 'best.pth')
            if platform.system() != 'Windows':
                mmcv.symlink(osp.basename(self.best_ckpt_path), dst_file)
            else:
                shutil.copy(osp.basename(self.best_ckpt_path), dst_file)


class DistEvalHook(BaseDistEvalHook):
    """Non-Distributed evaluation hook.

    Comparing with the ``EvalHook`` in MMCV, this hook will save the latest
    evaluation results as an attribute for other hooks to use (like
    `MMClsWandbHook`).
    """

    def __init__(self, dataloader, **kwargs):
        super(DistEvalHook, self).__init__(dataloader, **kwargs)

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
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

        results = self.test_fn(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            runner.logger.info("Dist Eval Hook : There are {} samples in total.".format(len(results)))
        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = {"acc": np.array(results).mean() * 100}

            for name, val in key_score.items():
                runner.log_buffer.output[name] = val
            runner.log_buffer.ready = True
            if self.save_best:
                if self.key_indicator == 'auto':
                    # infer from eval_results
                    self._init_rule(self.rule, list(key_score.keys())[0])
                self._save_ckpt(runner, key_score[self.key_indicator])

                dst_file = osp.join(self.out_dir, 'best.pth')
                if platform.system() != 'Windows':
                    mmcv.symlink(osp.basename(self.best_ckpt_path), dst_file)
                else:
                    shutil.copy(osp.basename(self.best_ckpt_path), dst_file)
