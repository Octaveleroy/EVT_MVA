import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from io import BytesIO
from typing import Union

import torch


def plot_histogram_gif(
    trajectories: Union[np.ndarray, torch.Tensor],
    output_path: str,
    bins: int = 50,
    fps: int = 10,
    frame_step: int = 1,
) -> None:
    """
    Create an animated GIF showing histogram evolution over time.

    Args:
        trajectories: Array of shape (M, N+1) from SDESimulator
        output_path: Path for output GIF file
        bins: Number of histogram bins
        fps: Frames per second in the GIF
        frame_step: Sample every N-th time step (reduces frames)
    """
    # Convert torch tensor to numpy if needed
    if isinstance(trajectories, torch.Tensor):
        trajectories = trajectories.cpu().numpy()

    M, T = trajectories.shape  # M trajectories, T time steps

    # Compute fixed bin edges from global range
    global_min = trajectories.min()
    global_max = trajectories.max()
    bin_edges = np.linspace(global_min, global_max, bins + 1)

    # Generate frames
    frames = []
    time_indices = range(0, T, frame_step)

    for t in time_indices:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(trajectories[:, t], bins=bin_edges, edgecolor="black", alpha=0.7)
        ax.set_xlim(global_min, global_max)
        ax.set_xlabel("X")
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution at time step {t}")

        # Save frame to buffer
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        frames.append(imageio.imread(buf))
        plt.close(fig)

    # Write GIF
    imageio.mimsave(output_path, frames, fps=fps)
