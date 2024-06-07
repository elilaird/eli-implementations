class LinearWarmupScheduler(object):
    def __init__(self, optimizer, warmup_steps, total_steps, start_lr, ref_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self._step = 0.0

    def step(self):
        self._step += 1
        if self._step <= self.warmup_steps:
            progress = float(self._step) / float(max(1, self.warmup_steps))
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        else:
            progress = float(self._step - self.warmup_steps) / float(
                max(1, self.total_steps - self.warmup_steps)
            )
            new_lr = self.ref_lr * (1 - progress)

        for group in self.optimizer.param_groups:
            group["lr"] = new_lr

        return new_lr


class MomentumScheduler:
    def __init__(self, ema, ipe, num_epochs, ipe_scale):
        self.ema = ema
        self.ipe = ipe
        self.num_epochs = num_epochs
        self.ipe_scale = ipe_scale
        self.iteration = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.iteration <= self.ipe * self.num_epochs * self.ipe_scale:
            momentum = self.ema[0] + self.iteration * (
                self.ema[1] - self.ema[0]
            ) / (self.ipe * self.num_epochs * self.ipe_scale)
            self.iteration += 1
            return momentum
        else:
            raise StopIteration


class WarmupCosineWDSchedule(object):
    """
    Adapted from: https://raw.githubusercontent.com/facebookresearch/ijepa/main/src/utils/schedulers.py
    """

    def __init__(
        self,
        optimizer,
        warmup_steps,
        start_wd,
        ref_wd,
        T_max,
        last_epoch=-1,
        final_wd=0.0,
    ):
        self.optimizer = optimizer
        self.start_wd = start_wd
        self.ref_wd = ref_wd
        self.final_wd = final_wd
        self.warmup_steps = warmup_steps
        self.T_max = T_max - warmup_steps
        self._step = 0.0

    def step(self):
        self._step += 1
        if self._step < self.warmup_steps:
            progress = float(self._step) / float(max(1, self.warmup_steps))
            new_wd = self.start_wd + progress * (self.ref_wd - self.start_wd)
        else:
            # -- progress after warmup
            progress = float(self._step - self.warmup_steps) / float(
                max(1, self.T_max)
            )
            new_wd = max(
                self.final_wd,
                self.final_wd
                + (self.ref_wd - self.final_wd)
                * 0.5
                * (1.0 + math.cos(math.pi * progress)),
            )

        for group in self.optimizer.param_groups:
            if ("WD_exclude" not in group) or not group["WD_exclude"]:
                group["weight_decay"] = new_wd

        return new_wd


class CosineWDSchedule(object):
    """
    Adapted from: https://raw.githubusercontent.com/facebookresearch/ijepa/main/src/utils/schedulers.py
    """

    def __init__(self, optimizer, ref_wd, T_max, final_wd=0.0):
        self.optimizer = optimizer
        self.ref_wd = ref_wd
        self.final_wd = final_wd
        self.T_max = T_max
        self._step = 0.0

    def step(self):
        self._step += 1
        progress = self._step / self.T_max
        new_wd = self.final_wd + (self.ref_wd - self.final_wd) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

        if self.final_wd <= self.ref_wd:
            new_wd = max(self.final_wd, new_wd)
        else:
            new_wd = min(self.final_wd, new_wd)

        for group in self.optimizer.param_groups:
            if ("WD_exclude" not in group) or not group["WD_exclude"]:
                group["weight_decay"] = new_wd
        return new_wd


class WarmupCosineSchedule(object):
    """
    Adapted from: https://raw.githubusercontent.com/facebookresearch/ijepa/main/src/utils/schedulers.py
    """

    def __init__(
        self,
        optimizer,
        warmup_steps,
        start_lr,
        ref_lr,
        T_max,
        last_epoch=-1,
        final_lr=0.0,
    ):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.T_max = T_max - warmup_steps
        self._step = 0.0

    def step(self):
        self._step += 1
        if self._step < self.warmup_steps:
            progress = float(self._step) / float(max(1, self.warmup_steps))
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        else:
            # -- progress after warmup
            progress = float(self._step - self.warmup_steps) / float(
                max(1, self.T_max)
            )
            new_lr = max(
                self.final_lr,
                self.final_lr
                + (self.ref_lr - self.final_lr)
                * 0.5
                * (1.0 + math.cos(math.pi * progress)),
            )

        for group in self.optimizer.param_groups:
            group["lr"] = new_lr

        return new_lr
