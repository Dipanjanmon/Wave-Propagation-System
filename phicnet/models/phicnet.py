"""
PhICNet Architecture and Experimental Baselines for Wave Propagation.

Combines explicit finite-difference physical operator with a residual encoder-decoder
source model (RED-Net) to perform physical state forecasting and unknown source recovery.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, List

from .physics_model import PhysicsWave2D
from .source_net import REDNet


class PhICNet(nn.Module):
    """
    PhICNet: Physics-Incorporated Convolutional Neural Network for 2nd order systems.
    
    Architecture (as per Section 4.3 of the paper):
    - Physics Branch: Discrete 2D wave equation operator.
    - Source Branch: RED-Net estimating unknown source V_{t+1} from C_{V,t} = [V_t, V_{t-1}].
    - Integration: \hat{U}_{t+1} = H_t + w_{vc} \odot C_{V,t} + g(C_{V,t})
    """
    def __init__(
        self,
        c: float = 2.0,
        dt: float = 0.005,
        dx: float = 1.0 / 63.0,
        hidden_dim: int = 32,
        non_negative_source: bool = False  # Wave sources can be negative
    ):
        super().__init__()
        self.c = c
        self.dt = dt
        self.dx = dx

        # Physics branch
        self.physics = PhysicsWave2D(c=c, dt=dt, dx=dx, learnable_c=False)

        # Source estimation branch (takes K=2 past discrepancies C_{V,t} = [V_t, V_{t-1}])
        self.source_net = REDNet(
            in_channels=2,
            out_channels=1,
            hidden_dim=hidden_dim,
            non_negative_source=non_negative_source
        )

        # w_vc initialized with coefficients of the finite difference approx of d^2/dt^2
        # For K=2, the coefficients are [2.0, -1.0] representing V_{t+1} ~ 2V_t - V_{t-1}
        self.w_vc = nn.Parameter(torch.tensor([2.0, -1.0]).view(1, 2, 1, 1))

    def compute_discrepancy(self, u_curr: torch.Tensor, u_prev: Optional[torch.Tensor], u_prev2: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Compute discrepancy between actual observation and physics-only prediction.
        For wave equation (2nd order): V_t = (U^t - U_{phy}^t) / dt^2
        """
        if u_prev is None or u_prev2 is None:
            # At t=0 or t=1, we don't have enough history to compute full discrepancy
            return torch.zeros_like(u_curr)
        
        u_phy_t = self.physics(u_prev, u_prev2)
        discrepancy = (u_curr - u_phy_t) / (self.dt ** 2)
        
        # Clamp discrepancy to avoid runaway feedback amplification
        discrepancy = torch.clamp(discrepancy, -500000.0, 500000.0)
        return discrepancy

    def step(
        self,
        u_curr: torch.Tensor,
        u_prev: Optional[torch.Tensor] = None,
        u_prev2: Optional[torch.Tensor] = None,
        disc_prev: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Advance one step:
        1. Physics: H_t = physics(u_curr, u_prev)
        2. Discrepancy: V_t = (u_curr - physics(u_prev, u_prev2)) / dt^2
        3. Source Memory: C_{V,t} = [V_t, V_{t-1}]
        4. Source: \hat{V}_{t+1} = w_{vc} \odot C_{V,t} + g(C_{V,t})
        5. Final: \hat{U}_{t+1} = H_t + dt^2 * \hat{V}_{t+1}
        """
        if u_prev is None:
            u_prev = torch.zeros_like(u_curr)
            
        # 1. Physics prediction H_t
        u_phy_next = self.physics(u_curr, u_prev)

        # 2. Extract current perturbation V_t
        disc_curr = self.compute_discrepancy(u_curr, u_prev, u_prev2)
        
        if disc_prev is None:
            disc_prev = torch.zeros_like(disc_curr)

        # 3. Source Memory C_{V,t}
        c_v = torch.cat([disc_curr, disc_prev], dim=1)

        # 4. Source Prediction \hat{V}_{t+1}
        w_term = (self.w_vc * c_v).sum(dim=1, keepdim=True)
        s_pred = w_term + self.source_net(c_v)

        # 5. Integrated prediction \hat{U}_{t+1}
        u_next = u_phy_next + (self.dt ** 2) * s_pred

        # Enforce boundary condition
        u_next[:, :, 0, :] = 0.0
        u_next[:, :, -1, :] = 0.0
        u_next[:, :, :, 0] = 0.0
        u_next[:, :, :, -1] = 0.0

        return u_next, s_pred, disc_curr

    def forward(
        self,
        u_seq: torch.Tensor,
        unroll_steps: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass over sequence of observed states for training.
        """
        if u_seq.dim() == 4:
            u_seq = u_seq.unsqueeze(2)

        b, t_steps, c, h, w = u_seq.shape
        u_preds = []
        s_preds = []
        u_phys = []

        disc_prev = None
        u_prev2 = None
        u_prev = None
        curr_u = u_seq[:, 0]

        for t in range(t_steps - 1):
            if unroll_steps <= 1 or (t % unroll_steps == 0):
                u_curr = u_seq[:, t]
                u_prev = u_seq[:, t - 1] if t > 0 else u_curr.clone() # Approx rest at t=0
                u_prev2 = u_seq[:, t - 2] if t > 1 else u_prev.clone()
            else:
                pass # already set at bottom of loop

            u_phy_next = self.physics(u_curr, u_prev)
            u_phys.append(u_phy_next)

            disc_curr = self.compute_discrepancy(u_curr, u_prev, u_prev2)
            if disc_prev is None:
                disc_prev = torch.zeros_like(disc_curr)
            
            c_v = torch.cat([disc_curr, disc_prev], dim=1)
            w_term = (self.w_vc * c_v).sum(dim=1, keepdim=True)
            s_pred = w_term + self.source_net(c_v)
            s_preds.append(s_pred)

            u_next = u_phy_next + (self.dt ** 2) * s_pred
            
            u_next[:, :, 0, :] = 0.0
            u_next[:, :, -1, :] = 0.0
            u_next[:, :, :, 0] = 0.0
            u_next[:, :, :, -1] = 0.0
            
            u_preds.append(u_next)

            disc_prev = disc_curr
            u_prev2 = u_prev
            u_prev = u_curr
            u_curr = u_next

        u_pred_seq = torch.stack(u_preds, dim=1)
        s_pred_seq = torch.stack(s_preds, dim=1)
        u_phy_seq = torch.stack(u_phys, dim=1)

        return u_pred_seq, s_pred_seq, u_phy_seq

    @torch.no_grad()
    def rollout_autoregressive(
        self,
        u_warmup: torch.Tensor,
        rollout_steps: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if u_warmup.dim() == 4:
            u_warmup = u_warmup.unsqueeze(2)

        b, t_warm, c, h, w = u_warmup.shape
        disc_prev = None
        u_prev2 = None
        u_prev = None

        for t in range(t_warm):
            u_curr = u_warmup[:, t]
            disc_curr = self.compute_discrepancy(u_curr, u_prev, u_prev2)
            
            if disc_prev is None:
                disc_prev = torch.zeros_like(disc_curr)
            c_v = torch.cat([disc_curr, disc_prev], dim=1)
            w_term = (self.w_vc * c_v).sum(dim=1, keepdim=True)
            s_pred = w_term + self.source_net(c_v)
            
            disc_prev = disc_curr
            u_prev2 = u_prev
            u_prev = u_curr

        u_preds = []
        s_preds = []
        
        # Need history to continue rollout autonomously
        u_prev2 = u_warmup[:, -3] if t_warm > 2 else (u_warmup[:, -2] if t_warm > 1 else u_warmup[:, -1])
        u_prev = u_warmup[:, -2] if t_warm > 1 else u_warmup[:, -1]
        curr_u = u_warmup[:, -1]

        for step in range(rollout_steps):
            u_phy_next = self.physics(curr_u, u_prev)
            disc_curr = self.compute_discrepancy(curr_u, u_prev, u_prev2)
            
            if disc_prev is None:
                disc_prev = torch.zeros_like(disc_curr)
            
            c_v = torch.cat([disc_curr, disc_prev], dim=1)
            w_term = (self.w_vc * c_v).sum(dim=1, keepdim=True)
            s_pred = w_term + self.source_net(c_v)
            s_preds.append(s_pred)

            next_u = u_phy_next + (self.dt ** 2) * s_pred
            
            next_u[:, :, 0, :] = 0.0
            next_u[:, :, -1, :] = 0.0
            next_u[:, :, :, 0] = 0.0
            next_u[:, :, :, -1] = 0.0
            u_preds.append(next_u)

            disc_prev = disc_curr
            u_prev2 = u_prev
            u_prev = curr_u
            curr_u = next_u

        return torch.stack(u_preds, dim=1), torch.stack(s_preds, dim=1)


class PhysicsOnlyBaseline(nn.Module):
    def __init__(self, c: float = 2.0, dt: float = 0.005, dx: float = 1.0 / 63.0):
        super().__init__()
        self.physics = PhysicsWave2D(c=c, dt=dt, dx=dx)

    def forward(self, u_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if u_seq.dim() == 4:
            u_seq = u_seq.unsqueeze(2)
        b, t_steps, c, h, w = u_seq.shape
        u_preds = []
        s_preds = []
        for t in range(t_steps - 1):
            u_curr = u_seq[:, t]
            u_prev = u_seq[:, t - 1] if t > 0 else u_curr
            u_next = self.physics(u_curr, u_prev)
            u_preds.append(u_next)
            s_preds.append(torch.zeros_like(u_curr))
        return torch.stack(u_preds, dim=1), torch.stack(s_preds, dim=1)

    @torch.no_grad()
    def rollout_autoregressive(self, u_init: torch.Tensor, rollout_steps: int) -> torch.Tensor:
        if u_init.dim() == 5:
            u_prev = u_init[:, -2] if u_init.shape[1] > 1 else u_init[:, -1]
            curr_u = u_init[:, -1]
        else:
            curr_u = u_init
            u_prev = u_init
            
        preds = []
        for _ in range(rollout_steps):
            next_u = self.physics(curr_u, u_prev)
            preds.append(next_u)
            u_prev = curr_u
            curr_u = next_u
        return torch.stack(preds, dim=1)


class CNNBaseline(nn.Module):
    """
    Directly predicts U^{t+1} from U^t and U^{t-1} using a CNN.
    """
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        # Takes U_curr and U_prev = 2 channels
        self.net = nn.Sequential(
            nn.Conv2d(2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1)
        )

    def forward(self, u_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if u_seq.dim() == 4:
            u_seq = u_seq.unsqueeze(2)
        b, t_steps, c, h, w = u_seq.shape
        u_preds = []
        s_preds = []
        for t in range(t_steps - 1):
            u_curr = u_seq[:, t]
            u_prev = u_seq[:, t - 1] if t > 0 else u_curr
            
            x = torch.cat([u_curr, u_prev], dim=1)
            u_next = self.net(x)
            
            u_next[:, :, 0, :] = 0.0
            u_next[:, :, -1, :] = 0.0
            u_next[:, :, :, 0] = 0.0
            u_next[:, :, :, -1] = 0.0
            u_preds.append(u_next)
            s_preds.append(torch.zeros_like(u_curr))
        return torch.stack(u_preds, dim=1), torch.stack(s_preds, dim=1)

    @torch.no_grad()
    def rollout_autoregressive(self, u_init: torch.Tensor, rollout_steps: int) -> torch.Tensor:
        if u_init.dim() == 5:
            u_prev = u_init[:, -2] if u_init.shape[1] > 1 else u_init[:, -1]
            curr_u = u_init[:, -1]
        else:
            curr_u = u_init
            u_prev = u_init

        preds = []
        for _ in range(rollout_steps):
            x = torch.cat([curr_u, u_prev], dim=1)
            next_u = self.net(x)
            
            next_u[:, :, 0, :] = 0.0
            next_u[:, :, -1, :] = 0.0
            next_u[:, :, :, 0] = 0.0
            next_u[:, :, :, -1] = 0.0
            
            preds.append(next_u)
            u_prev = curr_u
            curr_u = next_u
        return torch.stack(preds, dim=1)


class ConvLSTMBaseline(nn.Module):
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Takes U_curr and U_prev = 2 channels
        self.encoder = nn.Conv2d(2, hidden_dim, kernel_size=3, padding=1)
        self.conv_lstm = nn.ModuleList([
            nn.Conv2d(2 * hidden_dim, 4 * hidden_dim, kernel_size=3, padding=1)
        ])
        self.decoder = nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1)

    def forward(self, u_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if u_seq.dim() == 4:
            u_seq = u_seq.unsqueeze(2)
        b, t_steps, c, h, w = u_seq.shape
        u_preds = []
        s_preds = []

        h_state = torch.zeros(b, self.hidden_dim, h, w, device=u_seq.device)
        c_state = torch.zeros(b, self.hidden_dim, h, w, device=u_seq.device)

        for t in range(t_steps - 1):
            u_curr = u_seq[:, t]
            u_prev = u_seq[:, t - 1] if t > 0 else u_curr
            x = torch.cat([u_curr, u_prev], dim=1)
            
            feat = F.relu(self.encoder(x))
            
            combined = torch.cat([feat, h_state], dim=1)
            gates = self.conv_lstm[0](combined)
            i, f, o, g = torch.split(gates, self.hidden_dim, dim=1)
            c_state = torch.sigmoid(f) * c_state + torch.sigmoid(i) * torch.tanh(g)
            h_state = torch.sigmoid(o) * torch.tanh(c_state)

            u_next = self.decoder(h_state)
            u_next[:, :, 0, :] = 0.0
            u_next[:, :, -1, :] = 0.0
            u_next[:, :, :, 0] = 0.0
            u_next[:, :, :, -1] = 0.0
            u_preds.append(u_next)
            s_preds.append(torch.zeros_like(u_curr))

        return torch.stack(u_preds, dim=1), torch.stack(s_preds, dim=1)

    @torch.no_grad()
    def rollout_autoregressive(self, u_warmup: torch.Tensor, rollout_steps: int) -> torch.Tensor:
        if u_warmup.dim() == 4:
            u_warmup = u_warmup.unsqueeze(2)
        b, t_warm, c, h, w = u_warmup.shape

        h_state = torch.zeros(b, self.hidden_dim, h, w, device=u_warmup.device)
        c_state = torch.zeros(b, self.hidden_dim, h, w, device=u_warmup.device)

        for t in range(t_warm):
            u_curr = u_warmup[:, t]
            u_prev = u_warmup[:, t - 1] if t > 0 else u_curr
            x = torch.cat([u_curr, u_prev], dim=1)
            
            feat = F.relu(self.encoder(x))
            combined = torch.cat([feat, h_state], dim=1)
            gates = self.conv_lstm[0](combined)
            i, f, o, g = torch.split(gates, self.hidden_dim, dim=1)
            c_state = torch.sigmoid(f) * c_state + torch.sigmoid(i) * torch.tanh(g)
            h_state = torch.sigmoid(o) * torch.tanh(c_state)

        preds = []
        u_prev = u_warmup[:, -2] if t_warm > 1 else u_warmup[:, -1]
        curr_u = u_warmup[:, -1]
        
        for _ in range(rollout_steps):
            x = torch.cat([curr_u, u_prev], dim=1)
            feat = F.relu(self.encoder(x))
            combined = torch.cat([feat, h_state], dim=1)
            gates = self.conv_lstm[0](combined)
            i, f, o, g = torch.split(gates, self.hidden_dim, dim=1)
            c_state = torch.sigmoid(f) * c_state + torch.sigmoid(i) * torch.tanh(g)
            h_state = torch.sigmoid(o) * torch.tanh(c_state)
            
            next_u = self.decoder(h_state)
            next_u[:, :, 0, :] = 0.0
            next_u[:, :, -1, :] = 0.0
            next_u[:, :, :, 0] = 0.0
            next_u[:, :, :, -1] = 0.0
            preds.append(next_u)
            
            u_prev = curr_u
            curr_u = next_u

        return torch.stack(preds, dim=1)


if __name__ == "__main__":
    print("Testing PhICNet RED-Net architecture for Wave propagation...")
    dummy_seq = torch.randn(2, 10, 1, 64, 64)
    model = PhICNet(c=2.0, dt=0.005, dx=1.0/63.0)
    u_pred, s_pred, u_phy = model(dummy_seq)
    print(f"PhICNet sequence forward: u_pred {u_pred.shape}, s_pred {s_pred.shape}, u_phy {u_phy.shape}")
    assert u_pred.shape == (2, 9, 1, 64, 64)

    rollout_u, rollout_s = model.rollout_autoregressive(dummy_seq[:, :3], rollout_steps=5)
    print(f"PhICNet rollout: u {rollout_u.shape}, s {rollout_s.shape}")
    assert rollout_u.shape == (2, 5, 1, 64, 64)
    print("All PhICNet model checks passed!")
