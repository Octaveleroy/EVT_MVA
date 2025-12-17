import numpy as np
import torch
from typing import Optional, Union


class SDESimulator:
    """
    Vectorized SDE simulator using Euler-Maruyama method.

    Simulates the SDE: dX = drift(X) dt + diffusion(X) dW
    where drift(x) = -(A_0 + a * x) and diffusion(x) = sqrt(2 * (B_0 + b * x^2))
    """

    def __init__(
        self,
        M: int,
        N: int,
        Delta_t: float,
        X_0: float,
        backend: str = "numpy",
        device: str = "cpu",
        seed: Optional[int] = None,
    ):
        """
        Args:
            M: Number of trajectories
            N: Number of time steps
            Delta_t: Time step size
            X_0: Initial condition
            backend: "numpy" or "torch"
            device: For torch backend: "cpu" or "cuda"
            seed: Random seed for reproducibility
        """
        self.M = M
        self.N = N
        self.Delta_t = Delta_t
        self.sqrt_Delta_t = np.sqrt(Delta_t)
        self.X_0 = X_0
        self.backend = backend
        self.device = device
        if seed is not None:
            self.set_seed(seed)

    def set_seed(self, seed: int) -> None:
        """Reset random state for reproducibility."""
        if self.backend == "numpy":
            np.random.seed(seed)
        else:
            torch.manual_seed(seed)
            if self.device == "cuda":
                torch.cuda.manual_seed(seed)

    def simulate(
        self, A_0: float, a: float, B_0: float, b: float
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Run vectorized simulation.

        Args:
            A_0: Drift constant term
            a: Drift linear coefficient
            B_0: Diffusion constant term
            b: Diffusion quadratic coefficient

        Returns:
            Trajectories array of shape (M, N+1)
        """
        if self.backend == "numpy":
            return self._simulate_numpy(A_0, a, B_0, b)
        else:
            return self._simulate_torch(A_0, a, B_0, b)

    def _simulate_numpy(
        self, A_0: float, a: float, B_0: float, b: float
    ) -> np.ndarray:
        x = np.zeros((self.M, self.N + 1))
        x[:, 0] = self.X_0
        dW = np.random.normal(0.0, self.sqrt_Delta_t, size=(self.M, self.N))

        for j in range(self.N):
            drift = -(A_0 + a * x[:, j])
            diffusion = np.sqrt(2 * (B_0 + b * x[:, j] ** 2))
            x[:, j + 1] = x[:, j] + drift * self.Delta_t + diffusion * dW[:, j]
            x[:, j + 1] = np.abs(x[:, j + 1])

        return x

    def _simulate_torch(
        self, A_0: float, a: float, B_0: float, b: float
    ) -> torch.Tensor:
        x = torch.zeros((self.M, self.N + 1), device=self.device)
        x[:, 0] = self.X_0
        dW = torch.randn((self.M, self.N), device=self.device) * self.sqrt_Delta_t

        for j in range(self.N):
            drift = -(A_0 + a * x[:, j])
            diffusion = torch.sqrt(2 * (B_0 + b * x[:, j] ** 2))
            x[:, j + 1] = x[:, j] + drift * self.Delta_t + diffusion * dW[:, j]
            x[:, j + 1] = torch.abs(x[:, j + 1])

        return x

    def simulate_batch(
        self,
        A_0: Union[np.ndarray, "torch.Tensor"],
        a: Union[np.ndarray, "torch.Tensor"],
        B_0: Union[np.ndarray, "torch.Tensor"],
        b: Union[np.ndarray, "torch.Tensor"],
    ) -> Union[np.ndarray, "torch.Tensor"]:
        """
        Vectorized simulation across K parameter configurations.

        Args:
            A_0, a, B_0, b: Arrays of shape (K,) containing K parameter sets

        Returns:
            Trajectories of shape (K, M, N+1)
        """
        if self.backend == "numpy":
            return self._simulate_batch_numpy(A_0, a, B_0, b)
        else:
            return self._simulate_batch_torch(A_0, a, B_0, b)

    def _simulate_batch_numpy(
        self,
        A_0: np.ndarray,
        a: np.ndarray,
        B_0: np.ndarray,
        b: np.ndarray,
    ) -> np.ndarray:
        K = len(A_0)
        x = np.zeros((K, self.M, self.N + 1))
        x[:, :, 0] = self.X_0

        # Independent noise for each parameter set: (K, M, N)
        dW = np.random.normal(0.0, self.sqrt_Delta_t, size=(K, self.M, self.N))

        # Reshape params for broadcasting: (K, 1)
        A_0 = np.asarray(A_0).reshape(-1, 1)
        a = np.asarray(a).reshape(-1, 1)
        B_0 = np.asarray(B_0).reshape(-1, 1)
        b = np.asarray(b).reshape(-1, 1)

        for j in range(self.N):
            drift = -(A_0 + a * x[:, :, j])
            diffusion = np.sqrt(2 * (B_0 + b * x[:, :, j] ** 2))
            x[:, :, j + 1] = x[:, :, j] + drift * self.Delta_t + diffusion * dW[:, :, j]
            x[:, :, j + 1] = np.abs(x[:, :, j + 1])

        return x

    def _simulate_batch_torch(
        self,
        A_0: "torch.Tensor",
        a: "torch.Tensor",
        B_0: "torch.Tensor",
        b: "torch.Tensor",
    ) -> "torch.Tensor":
        # Convert to tensors if needed
        A_0 = torch.as_tensor(A_0, device=self.device, dtype=torch.float32)
        a = torch.as_tensor(a, device=self.device, dtype=torch.float32)
        B_0 = torch.as_tensor(B_0, device=self.device, dtype=torch.float32)
        b = torch.as_tensor(b, device=self.device, dtype=torch.float32)

        K = len(A_0)
        x = torch.zeros((K, self.M, self.N + 1), device=self.device)
        x[:, :, 0] = self.X_0

        dW = torch.randn((K, self.M, self.N), device=self.device) * self.sqrt_Delta_t

        # Reshape for broadcasting: (K, 1)
        A_0 = A_0.view(-1, 1)
        a = a.view(-1, 1)
        B_0 = B_0.view(-1, 1)
        b = b.view(-1, 1)

        for j in range(self.N):
            drift = -(A_0 + a * x[:, :, j])
            diffusion = torch.sqrt(2 * (B_0 + b * x[:, :, j] ** 2))
            x[:, :, j + 1] = x[:, :, j] + drift * self.Delta_t + diffusion * dW[:, :, j]
            x[:, :, j + 1] = torch.abs(x[:, :, j + 1])

        return x
