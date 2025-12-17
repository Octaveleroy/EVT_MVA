"""Training data generator for Neural Bayes Estimators.

Implements on-the-fly data generation following Sainsbury-Dale et al. (2024):
- Section 2.3.3: Fresh data simulated continuously during training
- Section 2.2.2 (Eq 7): Variable M ~ U[min_M, max_M] per batch
"""

import numpy as np
import torch
from typing import Dict, Tuple, Union, Iterator, Optional
from .simulator import SDESimulator
from .priors import Prior


class NeuralBayesDataGenerator:
    """
    Generate training data for Neural Bayes Estimators.

    Following the methodology in Sainsbury-Dale et al. (2024):
    - Sample K parameter vectors from prior
    - For each parameter, simulate M trajectories
    - Return (trajectories, parameters) pairs for training
    """

    def __init__(
        self,
        simulator: SDESimulator,
        priors: Dict[str, Prior],
        K: int = 10000,
    ):
        """
        Args:
            simulator: SDESimulator instance
            priors: Dict mapping param names to Prior objects
                    e.g., {'A_0': UniformPrior(0, 1), 'a': UniformPrior(0, 2), ...}
            K: Total number of parameter samples for training set
        """
        self.simulator = simulator
        self.priors = priors
        self.K = K
        self._param_names = ["A_0", "a", "B_0", "b"]

    def generate_dataset(
        self, n_params: int = None
    ) -> Tuple[
        Union[np.ndarray, torch.Tensor], Dict[str, Union[np.ndarray, torch.Tensor]]
    ]:
        """
        Generate a complete dataset.

        Args:
            n_params: Number of parameter sets (defaults to self.K)

        Returns:
            trajectories: Shape (n_params, M, N+1)
            params: Dict of parameter arrays, each shape (n_params,)
        """
        if n_params is None:
            n_params = self.K

        backend = self.simulator.backend
        device = self.simulator.device

        # Sample parameters from priors
        params = {}
        for name in self._param_names:
            if name in self.priors:
                params[name] = self.priors[name].sample(
                    n_params, backend=backend, device=device
                )
            else:
                raise KeyError(f"Prior for '{name}' not provided")

        # Simulate trajectories
        trajectories = self.simulator.simulate_batch(
            A_0=params["A_0"],
            a=params["a"],
            B_0=params["B_0"],
            b=params["b"],
        )

        return trajectories, params

    def generate_batches(
        self, batch_size: int, n_batches: int = None
    ) -> Iterator[
        Tuple[
            Union[np.ndarray, torch.Tensor], Dict[str, Union[np.ndarray, torch.Tensor]]
        ]
    ]:
        """
        Generate batches of training data (on-the-fly simulation).

        Args:
            batch_size: Number of parameter sets per batch
            n_batches: Number of batches (None = infinite)

        Yields:
            (trajectories, params) tuples
        """
        batch_count = 0
        while n_batches is None or batch_count < n_batches:
            yield self.generate_dataset(n_params=batch_size)
            batch_count += 1

    def to_pytorch_dataset(self, n_params: int = None) -> "SDEDataset":
        """
        Generate data and wrap in a PyTorch Dataset.

        Args:
            n_params: Number of parameter sets

        Returns:
            SDEDataset instance
        """
        trajectories, params = self.generate_dataset(n_params)
        return SDEDataset(trajectories, params)


class SDEDataset(torch.utils.data.Dataset):
    """PyTorch Dataset wrapper for SDE training data."""

    def __init__(
        self,
        trajectories: Union[np.ndarray, torch.Tensor],
        params: Dict[str, Union[np.ndarray, torch.Tensor]],
    ):
        """
        Args:
            trajectories: Shape (K, M, N+1)
            params: Dict of arrays, each shape (K,)
        """
        if isinstance(trajectories, np.ndarray):
            self.trajectories = torch.from_numpy(trajectories).float()
        else:
            self.trajectories = trajectories.float()

        # Stack parameters into tensor of shape (K, 4)
        param_list = []
        for name in ["A_0", "a", "B_0", "b"]:
            p = params[name]
            if isinstance(p, np.ndarray):
                p = torch.from_numpy(p).float()
            param_list.append(p)
        self.params = torch.stack(param_list, dim=1)

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            trajectories: Shape (M, N+1) - M trajectories for this parameter set
            params: Shape (4,) - [A_0, a, B_0, b]
        """
        return self.trajectories[idx], self.params[idx]


class OnTheFlyDataset(torch.utils.data.IterableDataset):
    """
    Infinite dataset that generates training data on-the-fly.

    Following Sainsbury-Dale et al. (2024):
    - Section 2.3.3: Fresh data simulated continuously during training
    - Section 2.2.2 (Eq 7): Variable M ~ U[min_M, max_M] per batch

    This prevents overfitting and allows using larger networks since data
    in stochastic-gradient-descent updates are always different.

    Args:
        simulator: SDESimulator instance (will use its N, Delta_t, X_0 settings)
        priors: Dict mapping param names to Prior objects
        batch_size: Number of samples to generate per iteration
        min_M: Minimum number of replicates (inclusive)
        max_M: Maximum number of replicates (inclusive)
        seed: Base random seed for reproducibility
    """

    def __init__(
        self,
        simulator: SDESimulator,
        priors: Dict[str, Prior],
        batch_size: int,
        min_M: int,
        max_M: int,
        seed: Optional[int] = None,
    ):
        self.simulator = simulator
        self.priors = priors
        self.batch_size = batch_size
        self.min_M = min_M
        self.max_M = max_M
        self.base_seed = seed
        self._param_names = ["A_0", "a", "B_0", "b"]

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Infinite iterator that yields (trajectories, params) batches.

        Each iteration:
        1. Samples M ~ U[min_M, max_M] for this batch
        2. Samples batch_size parameter vectors from priors
        3. Simulates trajectories for each parameter set
        4. Yields the entire batch (all samples share the same M)

        Yields:
            trajectories: Shape (batch_size, M, N+1) - batch of trajectory sets
            params: Shape (batch_size, 4) - [A_0, a, B_0, b] for each sample
        """
        # Per-worker seeding for DataLoader multiprocessing
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Different seed per worker to ensure different data
            worker_seed = (self.base_seed or 0) + worker_info.id * 1000
        else:
            worker_seed = self.base_seed

        rng = np.random.default_rng(worker_seed)

        while True:  # Infinite iterator
            # Sample M for this batch (Equation 7 from paper)
            M = int(rng.integers(self.min_M, self.max_M + 1))

            # Temporarily update simulator's M
            original_M = self.simulator.M
            self.simulator.M = M

            try:
                # Sample parameters from priors
                backend = self.simulator.backend
                device = self.simulator.device

                params = {}
                for name in self._param_names:
                    if name in self.priors:
                        params[name] = self.priors[name].sample(
                            self.batch_size, backend=backend, device=device
                        )
                    else:
                        raise KeyError(f"Prior for '{name}' not provided")

                # Simulate trajectories: shape (batch_size, M, N+1)
                trajectories = self.simulator.simulate_batch(
                    A_0=params["A_0"],
                    a=params["a"],
                    B_0=params["B_0"],
                    b=params["b"],
                )

                # Stack params into tensor: shape (batch_size, 4)
                param_list = []
                for name in self._param_names:
                    p = params[name]
                    if isinstance(p, np.ndarray):
                        p = torch.from_numpy(p).float()
                    elif not isinstance(p, torch.Tensor):
                        p = torch.tensor(p).float()
                    else:
                        p = p.float()
                    param_list.append(p)
                params_tensor = torch.stack(param_list, dim=1)

                # Convert trajectories to tensor
                if isinstance(trajectories, np.ndarray):
                    trajectories = torch.from_numpy(trajectories).float()
                else:
                    trajectories = trajectories.float()

                # Yield entire batch (all samples share the same M)
                yield trajectories, params_tensor

            finally:
                # Restore original M
                self.simulator.M = original_M
