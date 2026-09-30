"""
2D Finite Difference Wave Equation Solver with Unknown External Sources.

Equation:
    d^2U/dt^2 = c^2 * (d^2U/dx^2 + d^2U/dy^2) + S(x, y, t)

Numerical discretization:
    Central-difference in time and space.
    Stability requirement: r = c * dt / dx <= 1/sqrt(2) ~ 0.707 in 2D.
"""

import numpy as np
from typing import Tuple, Optional, Callable, Dict, Any


class WaveEquation2D:
    """
    2D Wave Equation numerical simulator using finite differences.
    """
    def __init__(
        self,
        grid_size: int = 64,
        domain_length: float = 1.0,
        c: float = 2.0,
        dt: float = 0.005,
        boundary_condition: str = "dirichlet"
    ):
        """
        Args:
            grid_size: Number of grid points along each spatial dimension (Nx = Ny).
            domain_length: Physical length of square domain [0, L] x [0, L].
            c: Wave propagation speed.
            dt: Time step size.
            boundary_condition: 'dirichlet' (zero displacement at boundaries) or 'neumann' (reflective).
        """
        self.nx = grid_size
        self.ny = grid_size
        self.l = domain_length
        self.c = c
        self.dt = dt
        self.boundary_condition = boundary_condition.lower()

        self.dx = domain_length / (grid_size - 1)
        self.dy = self.dx

        # Stability parameter check (Courant number)
        self.courant = self.c * self.dt / self.dx
        if self.courant > 0.707:
            raise ValueError(
                f"Stability criterion violated: Courant number = {self.courant:.4f} > 0.707. "
                f"Reduce dt (current: {dt}) or increase dx."
            )

        # Coordinate grid
        x = np.linspace(0, domain_length, self.nx)
        y = np.linspace(0, domain_length, self.ny)
        self.xx, self.yy = np.meshgrid(x, y)

    def compute_laplacian(self, u: np.ndarray) -> np.ndarray:
        """
        Compute discrete 2D Laplacian using 5-point stencil.
        """
        laplacian = np.zeros_like(u)
        
        # Interior points
        laplacian[1:-1, 1:-1] = (
            u[2:, 1:-1]
            + u[:-2, 1:-1]
            + u[1:-1, 2:]
            + u[1:-1, :-2]
            - 4.0 * u[1:-1, 1:-1]
        ) / (self.dx ** 2)

        # Boundary conditions
        if self.boundary_condition == "neumann":
            laplacian[0, 1:-1] = (2.0 * u[1, 1:-1] + u[0, 2:] + u[0, :-2] - 4.0 * u[0, 1:-1]) / (self.dx ** 2)
            laplacian[-1, 1:-1] = (2.0 * u[-2, 1:-1] + u[-1, 2:] + u[-1, :-2] - 4.0 * u[-1, 1:-1]) / (self.dx ** 2)
            laplacian[1:-1, 0] = (u[2:, 0] + u[:-2, 0] + 2.0 * u[1:-1, 1] - 4.0 * u[1:-1, 0]) / (self.dx ** 2)
            laplacian[1:-1, -1] = (u[2:, -1] + u[:-2, -1] + 2.0 * u[1:-1, -2] - 4.0 * u[1:-1, -1]) / (self.dx ** 2)

        return laplacian

    def step(self, u_curr: np.ndarray, u_prev: np.ndarray, source: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Advance state by one time step: 
        U^{t+1} = 2U^t - U^{t-1} + dt^2 * c^2 * laplacian(U^t) + dt^2 * source^t
        """
        laplacian = self.compute_laplacian(u_curr)
        u_next = 2.0 * u_curr - u_prev + (self.c ** 2) * (self.dt ** 2) * laplacian
        
        if source is not None:
            u_next = u_next + (self.dt ** 2) * source

        if self.boundary_condition == "dirichlet":
            u_next[0, :] = 0.0
            u_next[-1, :] = 0.0
            u_next[:, 0] = 0.0
            u_next[:, -1] = 0.0

        return u_next

    def simulate(
        self,
        u_init: np.ndarray,
        num_steps: int,
        source_func: Optional[Callable[[int, np.ndarray, np.ndarray], np.ndarray]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate wave equation over multiple time steps.

        Args:
            u_init: Initial displacement field of shape (nx, ny).
            num_steps: Number of simulation time steps.
            source_func: Function (t_idx, xx, yy) -> source_map of shape (nx, ny).

        Returns:
            u_trajectory: Array of shape (num_steps, nx, ny).
            s_trajectory: Array of shape (num_steps, nx, ny).
        """
        u_traj = np.zeros((num_steps, self.nx, self.ny), dtype=np.float32)
        s_traj = np.zeros((num_steps, self.nx, self.ny), dtype=np.float32)

        u_curr = u_init.copy().astype(np.float32)
        u_prev = u_curr.copy()  # Start from rest (du/dt = 0)

        if self.boundary_condition == "dirichlet":
            u_curr[0, :] = 0.0
            u_curr[-1, :] = 0.0
            u_curr[:, 0] = 0.0
            u_curr[:, -1] = 0.0
            u_prev = u_curr.copy()

        for t in range(num_steps):
            u_traj[t] = u_curr
            if source_func is not None:
                s_curr = source_func(t, self.xx, self.yy).astype(np.float32)
            else:
                s_curr = np.zeros_like(u_curr, dtype=np.float32)
            s_traj[t] = s_curr
            
            u_next = self.step(u_curr, u_prev, s_curr)
            u_prev = u_curr
            u_curr = u_next

        return u_traj, s_traj


def make_gaussian_source(
    xx: np.ndarray,
    yy: np.ndarray,
    center_x: float,
    center_y: float,
    sigma: float = 0.05,
    amplitude: float = 10.0
) -> np.ndarray:
    """
    Generate a localized 2D Gaussian perturbation.
    """
    r_sq = (xx - center_x) ** 2 + (yy - center_y) ** 2
    return amplitude * np.exp(-r_sq / (2.0 * sigma ** 2))


def generate_wave_trajectory(
    grid_size: int = 64,
    num_steps: int = 100,
    c: float = 2.0,
    dt: float = 0.005,
    source_type: str = "coupled_oscillators",
    rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generate a sequence of wave propagation with unknown sources.
    
    Source types:
        - 'coupled_oscillators': Two localized sources oscillating over time.
        - 'single_oscillator': One localized source oscillating.
        - 'pulse': A single brief impact.
        - 'none': Free wave propagation.
    """
    if rng is None:
        rng = np.random.default_rng()

    solver = WaveEquation2D(
        grid_size=grid_size,
        domain_length=1.0,
        c=c,
        dt=dt,
        boundary_condition="dirichlet"
    )

    u_init = np.zeros((grid_size, grid_size), dtype=np.float32)
    source_params = {}

    if source_type == "coupled_oscillators":
        cx1, cy1 = rng.uniform(0.2, 0.4), rng.uniform(0.2, 0.8)
        cx2, cy2 = rng.uniform(0.6, 0.8), rng.uniform(0.2, 0.8)
        sigma = 0.03
        amp = rng.uniform(50000.0, 100000.0)
        freq1 = rng.uniform(10.0, 20.0)
        freq2 = rng.uniform(10.0, 20.0)
        
        source_params = {"cx1": cx1, "cy1": cy1, "cx2": cx2, "cy2": cy2, "freq1": freq1, "freq2": freq2}

        def source_fn(t_idx, xx, yy):
            time_sec = t_idx * dt
            s1 = make_gaussian_source(xx, yy, cx1, cy1, sigma, amp * np.sin(2 * np.pi * freq1 * time_sec))
            s2 = make_gaussian_source(xx, yy, cx2, cy2, sigma, amp * np.sin(2 * np.pi * freq2 * time_sec + np.pi/4))
            return s1 + s2

    elif source_type == "single_oscillator":
        cx, cy = rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7)
        sigma = 0.03
        amp = rng.uniform(50000.0, 100000.0)
        freq = rng.uniform(10.0, 20.0)
        
        source_params = {"cx": cx, "cy": cy, "freq": freq}

        def source_fn(t_idx, xx, yy):
            time_sec = t_idx * dt
            return make_gaussian_source(xx, yy, cx, cy, sigma, amp * np.sin(2 * np.pi * freq * time_sec))

    elif source_type == "pulse":
        cx, cy = rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7)
        sigma = 0.03
        amp = rng.uniform(100000.0, 300000.0)
        
        source_params = {"cx": cx, "cy": cy}

        def source_fn(t_idx, xx, yy):
            if t_idx < 10:
                return make_gaussian_source(xx, yy, cx, cy, sigma, amp)
            return np.zeros_like(xx)

    elif source_type == "none":
        # Initial condition perturbation instead of continuous source
        cx, cy = rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7)
        u_init = make_gaussian_source(solver.xx, solver.yy, cx, cy, 0.05, 1.0).astype(np.float32)
        source_fn = None
    else:
        raise ValueError(f"Unknown source_type: {source_type}")

    u_traj, s_traj = solver.simulate(u_init, num_steps, source_fn)
    metadata = {
        "c": c,
        "dt": dt,
        "grid_size": grid_size,
        "num_steps": num_steps,
        "source_type": source_type,
        "source_params": source_params
    }
    return u_traj, s_traj, metadata

if __name__ == "__main__":
    print("Testing 2D Wave Equation Solver...")
    u, s, meta = generate_wave_trajectory(
        grid_size=64, num_steps=50, c=300.0, dt=0.001, source_type="coupled_oscillators"
    )
    print(f"Generated trajectory shape: U = {u.shape}, S = {s.shape}")
    print(f"U min: {u.min():.4f}, max: {u.max():.4f}, mean: {u.mean():.4f}")
    assert not np.isnan(u).any(), "NaN found in U trajectory!"
    assert not np.isnan(s).any(), "NaN found in S trajectory!"
    print("Test passed successfully!")
