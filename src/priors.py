"""Prior distributions for Neural Bayes Estimator training."""

import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Union


class Prior(ABC):
    """Abstract base class for prior distributions."""

    @abstractmethod
    def sample(
        self, n: int, backend: str = "numpy", device: str = "cpu"
    ) -> Union[np.ndarray, torch.Tensor]:
        """Sample n values from the prior."""
        pass


class UniformPrior(Prior):
    """Uniform prior on [low, high]."""

    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high

    def sample(
        self, n: int, backend: str = "numpy", device: str = "cpu"
    ) -> Union[np.ndarray, torch.Tensor]:
        if backend == "numpy":
            return np.random.uniform(self.low, self.high, size=n)
        else:
            samples = torch.rand(n, device=device)
            return samples * (self.high - self.low) + self.low


class LogUniformPrior(Prior):
    """Log-uniform (Jeffreys) prior: log(x) ~ Uniform(log(low), log(high))."""

    def __init__(self, low: float, high: float):
        self.log_low = np.log(low)
        self.log_high = np.log(high)

    def sample(
        self, n: int, backend: str = "numpy", device: str = "cpu"
    ) -> Union[np.ndarray, torch.Tensor]:
        if backend == "numpy":
            log_samples = np.random.uniform(self.log_low, self.log_high, size=n)
            return np.exp(log_samples)
        else:
            log_samples = torch.rand(n, device=device)
            log_samples = log_samples * (self.log_high - self.log_low) + self.log_low
            return torch.exp(log_samples)


class NormalPrior(Prior):
    """Truncated normal prior."""

    def __init__(
        self, mean: float, std: float, low: float = None, high: float = None
    ):
        self.mean = mean
        self.std = std
        self.low = low
        self.high = high

    def sample(
        self, n: int, backend: str = "numpy", device: str = "cpu"
    ) -> Union[np.ndarray, torch.Tensor]:
        if backend == "numpy":
            samples = np.random.normal(self.mean, self.std, size=n)
            if self.low is not None:
                samples = np.maximum(samples, self.low)
            if self.high is not None:
                samples = np.minimum(samples, self.high)
            return samples
        else:
            samples = torch.randn(n, device=device) * self.std + self.mean
            if self.low is not None:
                samples = torch.clamp(samples, min=self.low)
            if self.high is not None:
                samples = torch.clamp(samples, max=self.high)
            return samples
