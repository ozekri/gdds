"""Base interface for Semantic-Informed Kernels."""

from __future__ import annotations
from typing import Literal, Optional, Tuple
import torch

class SIKKernel(torch.nn.Module):
    """Abstract base class for Semantic-Informed Kernels."""
    
    def __init__(
        self,
        embeddings: torch.Tensor,
        epsilon: float = 0.01,
        gamma: float = 0.0,
        metric: Literal["gaussian", "cosine"] = "gaussian",
        variable_bandwidth: bool = True,
        k_neighbors: int = 7,
    ):
        super().__init__()
        self.embeddings = embeddings.float()
        self.m = self.embeddings.shape[0]
        self.epsilon = epsilon
        self.gamma = gamma
        self.metric = metric
        self.variable_bandwidth = variable_bandwidth
        self.k_neighbors = k_neighbors
        self.device = self.embeddings.device

    def get_logR_column(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Get the log-normalized kernel values for given source tokens.
        
        Args:
            token_ids: Source token indices [N].
            
        Returns:
            logR: [N, m] dense or specialized sparse/lazy representation.
        """
        raise NotImplementedError

    def sample_neighbors(self, token_ids: torch.Tensor, exponent: torch.Tensor) -> torch.Tensor:
        """Sample next tokens from the kernel for given sources.
        
        Args:
            token_ids: Current token indices [N].
            exponent: alpha(t)^beta values [N].
            
        Returns:
            next_tokens: [N] sampled indices.
        """
        raise NotImplementedError

    def _compute_local_bandwidths(self) -> torch.Tensor:
        """Compute sigma_i for each token."""
        raise NotImplementedError
