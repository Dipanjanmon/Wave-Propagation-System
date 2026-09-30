"""
Composite Physics and Source Losses for PhICNet Wave System.

Loss formulation:
    L = L_pred + beta * L_source + lambda_sparse * L_sparse

Where:
- L_pred: Prediction error between predicted physical state U_pred and true observation U_true.
- L_source: Source consistency loss enforcing agreement between learned source S_hat and
            the physical discrepancy (d^2U/dt^2 - c^2 * Laplacian(U)).
- L_sparse: L1 sparsity penalty enforcing spatial localization of the recovered source.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


def compute_relative_l2_error(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Compute relative L2 error: ||pred - target||_2 / (||target||_2 + eps)
    """
    diff_norm = torch.norm(pred - target, p=2, dim=(-2, -1))
    target_norm = torch.norm(target, p=2, dim=(-2, -1))
    return torch.mean(diff_norm / (target_norm + eps))


class PhICNetLoss(nn.Module):
    """
    Multi-term loss function for PhICNet wave equation.
    """
    def __init__(
        self,
        beta_source: float = 0.1,
        lambda_sparse: float = 0.001,
        c: float = 2.0,
        dt: float = 0.005,
        dx: float = 1.0 / 63.0
    ):
        super().__init__()
        self.beta_source = beta_source
        self.lambda_sparse = lambda_sparse
        self.c = c
        self.dt = dt
        self.dx = dx

        laplacian_weight = torch.tensor([
            [[[0.0,  1.0, 0.0],
              [1.0, -4.0, 1.0],
              [0.0,  1.0, 0.0]]]
        ], dtype=torch.float32) / (dx ** 2)
        self.register_buffer("laplacian_kernel", laplacian_weight)

    def compute_laplacian(self, u: torch.Tensor) -> torch.Tensor:
        u_padded = F.pad(u, (1, 1, 1, 1), mode="constant", value=0.0)
        laplacian = F.conv2d(u_padded, self.laplacian_kernel, padding=0)
        laplacian[:, :, 0, :] = 0.0
        laplacian[:, :, -1, :] = 0.0
        laplacian[:, :, :, 0] = 0.0
        laplacian[:, :, :, -1] = 0.0
        return laplacian

    def forward(
        self,
        u_pred: torch.Tensor,
        u_true: torch.Tensor,
        s_pred: torch.Tensor,
        u_curr: torch.Tensor,
        u_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            u_pred: Model prediction at next time step (B, T, 1, H, W)
            u_true: Ground truth state at next time step (B, T, 1, H, W)
            s_pred: Model predicted source map (B, T, 1, H, W)
            u_curr: Current state observation (B, T, 1, H, W)
            u_prev: Previous state observation (B, T, 1, H, W)
        """
        # 1. State prediction loss (MSE)
        l_pred = F.mse_loss(u_pred, u_true)

        # 2. Source consistency loss
        # Physical residual: S_res = (U^{t+1} - 2U^t + U^{t-1}) / dt^2 - c^2 * Laplacian(U^t)
        b, t, ch, h, w = u_curr.shape
        u_curr_flat = u_curr.reshape(-1, ch, h, w)
        laplacian_curr = self.compute_laplacian(u_curr_flat).reshape(b, t, ch, h, w)

        s_res = (u_true - 2.0 * u_curr + u_prev) / (self.dt ** 2) - (self.c ** 2) * laplacian_curr
        
        # Zero boundaries where finite-difference stencil has boundary effects
        s_res_interior = s_res[..., 2:-2, 2:-2]
        s_pred_interior = s_pred[..., 2:-2, 2:-2]
        
        # Wave source is not strictly positive, so enforce MSE without ReLU
        l_source = F.mse_loss(s_pred_interior, s_res_interior)

        # 3. Sparsity loss (L1 norm on predicted source)
        l_sparse = torch.mean(torch.abs(s_pred))

        # Total combined loss
        total_loss = l_pred + self.beta_source * l_source + self.lambda_sparse * l_sparse

        metrics = {
            "loss_total": total_loss.item(),
            "loss_pred": l_pred.item(),
            "loss_source": l_source.item(),
            "loss_sparse": l_sparse.item(),
            "rel_l2_u": compute_relative_l2_error(u_pred, u_true).item()
        }

        return total_loss, metrics
