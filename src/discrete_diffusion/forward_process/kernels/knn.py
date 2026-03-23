"""k-NN (Sparse) implementation of the SIK Kernel."""

from __future__ import annotations
from typing import Literal, Optional, Tuple
import torch
from .base import SIKKernel

class KNNKernel(SIKKernel):
    """SIK Kernel using an exact k-NN graph for efficient sparse transitions."""
    
    def __init__(
        self,
        embeddings: torch.Tensor,
        epsilon: float = 0.01,
        gamma: float = 0.0,
        metric: Literal["gaussian", "cosine"] = "gaussian",
        variable_bandwidth: bool = True,
        k_neighbors: int = 7,
        top_k: int = 64,
        degree_chunk_size: int = 1000,
    ):
        super().__init__(embeddings, epsilon, gamma, metric, variable_bandwidth, k_neighbors)
        self.top_k = top_k
        self.degree_chunk_size = degree_chunk_size
        
        if metric == "cosine":
            # Normalize for cosine distance
            self.embeddings = self.embeddings / self.embeddings.norm(dim=1, keepdim=True).clamp(min=1e-10)

        self._sigma = None
        if variable_bandwidth:
            self._sigma = self._compute_local_bandwidths()

        # Pre-compute normalization factors (n_i)
        # MUST sum over full vocabulary if gamma > 0 to remain mathematically exact
        self._n = None
        if gamma > 0:
            self._n = self._compute_n_exact()

        self._knn_indices = None
        self._knn_distances = None
        self._logR_vocab = None
        self._build_knn_graph()
        self._logR_vocab = self._build_logR_vocab()

    def _compute_local_bandwidths(self) -> torch.Tensor:
        """σ_i per token (k-th neighbor distance). [m]. Chunked to avoid OOM."""
        with torch.no_grad():
            E = self.embeddings
            sigmas = torch.zeros(self.m, device=self.device, dtype=self.embeddings.dtype)
            cs = self.degree_chunk_size
            
            for start in range(0, self.m, cs):
                end = min(start + cs, self.m)
                chunk = E[start:end]
                
                if self.metric == "gaussian":
                    D_chunk = torch.cdist(chunk, E, p=2).pow(2)
                else:
                    D_chunk = 1.0 - torch.mm(chunk, E.t())
                
                # Distance to k-th neighbor
                D_chunk = D_chunk + 1e-10
                vals, _ = D_chunk.topk(self.k_neighbors + 1, dim=1, largest=False)
                sigmas[start:end] = vals[:, self.k_neighbors]
                
            return sigmas.clamp(min=1e-10)

    def _compute_n_exact(self) -> torch.Tensor:
        """d_i = Σ_j K(i,j). [m]. Mathematically exact normalization."""
        device = self.embeddings.device
        d = torch.zeros(self.m, device=device, dtype=self.embeddings.dtype)
        cs = self.degree_chunk_size
        for start in range(0, self.m, cs):
            end = min(start + cs, self.m)
            src_chunk = self.embeddings[start:end]
            
            if self.metric == "gaussian":
                dist = torch.cdist(src_chunk, self.embeddings, p=2).pow(2)
            else:
                dist = 1.0 - torch.mm(src_chunk, self.embeddings.t())
            
            if self.variable_bandwidth:
                bw = self._sigma[start:end].unsqueeze(1) * self._sigma.unsqueeze(0)
                K_chunk = torch.exp(-dist / (self.epsilon * bw))
            else:
                K_chunk = torch.exp(-dist / self.epsilon)
            d[start:end] = K_chunk.sum(dim=1)
        return d.pow(self.gamma)

    def _build_knn_graph(self) -> None:
        """Build exact k-NN graph using normalized distance metric."""
        try:
            from tqdm import tqdm
            pbar = tqdm(total=self.m, desc="[SIK] Building k-NN graph")
            use_tqdm = True
        except ImportError:
            use_tqdm = False

        k = min(self.top_k, self.m - 1)
        self._knn_indices = torch.zeros(self.m, k, dtype=torch.long, device=self.device)
        self._knn_distances = torch.zeros(self.m, k, dtype=self.embeddings.dtype, device=self.device)
        
        cs = self.degree_chunk_size
        with torch.no_grad():
            for start in range(0, self.m, cs):
                end = min(start + cs, self.m)
                src_chunk = self.embeddings[start:end]
                
                if self.metric == "gaussian":
                    D_chunk = torch.cdist(src_chunk, self.embeddings, p=2).pow(2)
                else:
                    D_chunk = 1.0 - torch.mm(src_chunk, self.embeddings.t())
                
                # Mask self
                row_idx = torch.arange(end - start, device=self.device)
                global_idx = torch.arange(start, end, device=self.device)
                D_chunk[row_idx, global_idx] = float("inf")
                
                # If variable bandwidth, normalize distance before top-k
                if self.variable_bandwidth:
                    bw = (self._sigma[start:end].unsqueeze(1) * self._sigma.unsqueeze(0)).clamp(min=1e-10)
                    D_chunk = D_chunk / bw
                
                vals, idx = D_chunk.topk(k, dim=1, largest=False)
                self._knn_indices[start:end] = idx
                self._knn_distances[start:end] = vals
                if use_tqdm: pbar.update(end - start)
        if use_tqdm: pbar.close()

    def _build_logR_vocab(self) -> torch.Tensor:
        """Precompute logR for all vocab tokens: [V, k] bf16."""
        V, k = self._knn_indices.shape
        knn_dist = self._knn_distances
        K_vals = torch.exp(-knn_dist / self.epsilon)
        
        n_i = self._n if self._n is not None else torch.ones(V, device=self.device)
        n_j = n_i[self._knn_indices]
        R_vals = K_vals / (n_i.unsqueeze(1) * n_j).clamp(min=1e-10)
        
        # Mask diagonal (if it somehow ended up in top-k)
        row_idx = torch.arange(V, device=self.device).unsqueeze(1).expand(V, k)
        diagonal_mask = self._knn_indices == row_idx
        R_vals = R_vals.clone(); R_vals[diagonal_mask] = 0.0

        logR = torch.full_like(R_vals, float("-inf"), dtype=torch.float32)
        mask = R_vals > 0
        logR[mask] = torch.log(R_vals[mask])
        return logR.to(torch.bfloat16)

    def sample_neighbors(self, token_ids: torch.Tensor, exponent: torch.Tensor) -> torch.Tensor:
        curr_logR = self._logR_vocab[token_ids]
        curr_knn = self._knn_indices[token_ids]
        logits = exponent.unsqueeze(-1) * curr_logR.float()
        gumbel = -torch.log(-torch.log(torch.rand_like(logits).clamp(min=1e-10)))
        neighbor_idx = torch.argmax(logits + gumbel, dim=-1)
        return torch.gather(curr_knn, -1, neighbor_idx.unsqueeze(-1)).squeeze(-1)
