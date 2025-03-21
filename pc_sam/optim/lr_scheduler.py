# https://github.com/facebookresearch/detectron2/blob/92ae9f0b92aba5867824b4f12aa06a22a60a45d3/detectron2/solver/lr_scheduler.py
# In fact, pytorch also has `SequentialLR`: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.SequentialLR.html
from bisect import bisect_right
from typing import List

import torch
from torch.optim.lr_scheduler import LRScheduler


class WarmupMultiStepLR(LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        milestones: List[int],
        gamma: float = 0.1,
        warmup_factor: float = 0.001,
        warmup_iters: int = 1000,
        warmup_method: str = "linear",
        last_epoch: int = -1,
    ):
        # logger.warning(
        #     "WarmupMultiStepLR is deprecated! Use LRMultipilier with fvcore ParamScheduler instead!"
        # )
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
        )
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()


def _get_warmup_factor_at_iter(
    method: str, iter: int, warmup_iters: int, warmup_factor: float
) -> float:
    """
    Return the learning rate warmup factor at a specific iteration.
    See :paper:`ImageNet in 1h` for more details.

    Args:
        method (str): warmup method; either "constant" or "linear".
        iter (int): iteration at which to calculate the warmup factor.
        warmup_iters (int): the number of warmup iterations.
        warmup_factor (float): the base warmup factor (the meaning changes according
            to the method used).

    Returns:
        float: the effective warmup factor at the given iteration.
    """
    if iter >= warmup_iters:
        return 1.0

    if method == "constant":
        return warmup_factor
    elif method == "linear":
        alpha = iter / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    else:
        raise ValueError("Unknown warmup method: {}".format(method))


def main():
    optimizer = torch.optim.SGD([torch.rand(1)], lr=0.1)
    scheduler = WarmupMultiStepLR(optimizer, milestones=[30, 60], warmup_iters=10)
    for epoch in range(100):
        optimizer.step()
        print(epoch, scheduler.get_last_lr())
        scheduler.step()


if __name__ == "__main__":
    main()
