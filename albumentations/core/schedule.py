import math
import random

import numpy as np

__all__ = ["LinearIncreaseSchedule", "CosineWarmupAndDecaySchedule"]




class UniformSampler:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def value(self):
        return random.uniform(float(self.a), float(self.b))

    def __float__(self):
        return self.value()


class LinearIncreaseSchedule:
    current_step: int = 0
    start_p: float = 0.0
    final_p: float = 1.0

    def __init__(self, num_steps: int, start_p=0.0, final_p: float = 0.5):
        self.current_step = 0
        self.num_steps = num_steps
        self.start_p = start_p
        self.final_p = final_p

    def reset(self):
        self.current_step = 0

    def step(self, step: int = None):
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1

    def __float__(self):
        return np.interp(
            float(self.current_step) / float(self.num_steps),
            [0, 1],
            [self.start_p, self.final_p],
            left=self.start_p,
            right=self.final_p,
        )


class CosineWarmupAndDecaySchedule:
    current_step: int = 0
    start_p: float = 0.0
    maximum_p: float = 0.5
    final_p: float = 1.0
    warmup_steps: int
    decay_steps: int

    def __init__(
        self,
        num_steps: int,
        start_p: float = 0.0,
        maximum_p: float = 0.5,
        final_p: float = 0.25,
        peak_fraction: float = 0.5,
    ):
        self.current_step = 0
        self.num_steps = num_steps
        self.warmup_steps = int(peak_fraction * num_steps + 0.5)
        self.decay_steps = num_steps - self.warmup_steps
        self.start_p = start_p
        self.maximum_p = maximum_p
        self.final_p = final_p

    def reset(self):
        self.current_step = 0

    def step(self, step: int = None) -> float:
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
        return self.value()

    def value(self) -> float:
        if self.current_step > self.warmup_steps:
            progress = (self.current_step - self.warmup_steps) / float(self.decay_steps)
            alpha = (math.cos(math.pi * progress) + 1) / 2
            return self.final_p + (self.maximum_p - self.final_p) * alpha
        else:
            progress = self.current_step / float(self.warmup_steps)
            alpha = (1 - math.cos(math.pi * progress)) / 2
            return self.start_p + (self.maximum_p - self.start_p) * alpha

    def __float__(self):
        return self.value()
