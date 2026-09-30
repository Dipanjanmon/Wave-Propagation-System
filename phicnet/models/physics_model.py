"""
Physics-Incorporated Operator Module for 2D Wave Propagation.

Implements the differential Laplacian operator using 2D convolutions with
finite-difference stencil kernels to solve the second-order wave equation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsWave2D(nn.Module):
    """
    Differentiable 2D Wave Equation operator using fixed 3x3 finite-difference convolution kernels.
    
    Wave update:
        U_{phy}^{t+1} = 2U^t - U^{t-1} + dt^2 * c^2 * (d^2 U / dx^2 + d^2 U / dy^2)
    """
    def __init__(
        self,
        c: float = 2.0,
        dt: float = 0.005,
        dx: float = 1.0 / 63.0,
        learnable_c: bool = False
    ):
        super().__init__()
        self.dt = dt
        self.dx = dx

        if learnable_c:
            self.c = nn.Parameter(torch.tensor(c, dtype=torch.float32))
        else:
            self.register_buffer("c", torch.tensor(c, dtype=torch.float32))

        # Combined 5-point discrete Laplacian kernel:
        laplacian_weight = torch.tensor([
            [[[0.0,  1.0, 0.0],
              [1.0, -4.0, 1.0],
              [0.0,  1.0, 0.0]]]
        ], dtype=torch.float32) / (dx ** 2)

        self.register_buffer("laplacian_kernel", laplacian_weight)

    def compute_laplacian(self, u: torch.Tensor) -> torch.Tensor:
        """
        Compute discrete Laplacian for batch u of shape (B, 1, H, W).
        Uses Dirichlet zero-boundary padding.
        """
        is_3d = False
        if u.dim() == 2:
            u = u.unsqueeze(0).unsqueeze(0)
        elif u.dim() == 3:
            u = u.unsqueeze(1)
            is_3d = True

        u_padded = F.pad(u, (1, 1, 1, 1), mode="constant", value=0.0)
        laplacian = F.conv2d(u_padded, self.laplacian_kernel, padding=0)
        
        # Enforce Dirichlet zero boundary on Laplacian output
        laplacian[:, :, 0, :] = 0.0
        laplacian[:, :, -1, :] = 0.0
        laplacian[:, :, :, 0] = 0.0
        laplacian[:, :, :, -1] = 0.0

        if is_3d:
            laplacian = laplacian.squeeze(1)

        return laplacian

    def forward(self, u_curr: torch.Tensor, u_prev: torch.Tensor) -> torch.Tensor:
        """
        Compute physics-only next-state prediction U_{phy}^{t+1}:
            U_{phy}^{t+1} = 2U^t - U^{t-1} + dt^2 * c^2 * Laplacian(U^t)
        """
        orig_dim = u_curr.dim()
        if orig_dim == 3:
            u_curr = u_curr.unsqueeze(1)
            u_prev = u_prev.unsqueeze(1)

        laplacian = self.compute_laplacian(u_curr)
        u_next = 2.0 * u_curr - u_prev + (self.c ** 2) * (self.dt ** 2) * laplacian
        
        # Enforce boundary condition
        u_next[:, :, 0, :] = 0.0
        u_next[:, :, -1, :] = 0.0
        u_next[:, :, :, 0] = 0.0
        u_next[:, :, :, -1] = 0.0

        if orig_dim == 3:
            u_next = u_next.squeeze(1)

        return u_next


if __name__ == "__main__":
    phy = PhysicsWave2D(c=300.0, dt=0.001, dx=1.0/63.0)
    u_c = torch.randn(2, 1, 64, 64)
    u_p = torch.randn(2, 1, 64, 64)
    out = phy(u_c, u_p)
    print(f"Physics operator test: input shape {u_c.shape} -> output shape {out.shape}")
    assert out.shape == u_c.shape, "Shape mismatch!"
    print("Physics operator test passed successfully.")
