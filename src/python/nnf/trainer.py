# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torch import nn
import numpy as np
import copy

from .runner import Runner


class ModelWithLoss(nn.Module):
    def __init__(self, model, loss_func):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.loss_func = loss_func

    def forward(self, data, target):
        output = self.model(data)
        loss = self.loss_func(output, target)
        return loss


class Trainer(object):
    def __init__(self, model, loss_func=None, device="cuda:0", codegen_flags=None, **kwargs):
        super(Trainer, self).__init__()
        self.model = model
        self.loss_func = loss_func
        if self.loss_func:
            self.model_with_loss = ModelWithLoss(self.model,
                                                 self.loss_func).to(device)
        else:
            self.model_with_loss = model
        self.device = device
        trainer_flags = {
            "autodiff": 1,  # add backward graph
            "training_mode": 1,  # move weight external
            "extern_result_memory": 1  # move result external
        }
        self._codegen_flags = copy.deepcopy(codegen_flags) or {}
        self._codegen_flags.update(trainer_flags)
        self.runner = Runner(self.model_with_loss,
                             codegen_flags=self._codegen_flags,
                             **kwargs)

    def __call__(self, *args):
        return self.run_by_nnf(*args)

    def run_by_pytorch(self, *args):
        return self.model_with_loss(*args)

    def run_by_nnf(self, *args):
        outs = self.runner(*args)
        for out in outs:
            if np.prod(out.shape) == 1:
                return out
        assert False