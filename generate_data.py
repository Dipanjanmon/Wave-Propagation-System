"""
Generate synthetic 2D wave propagation dataset with unknown external sources.

Saves train, val, and test splits as compressed NPZ or NPY archives containing:
- 'u': (N_sequences, seq_len, H, W)
- 's': (N_sequences, seq_len, H, W)
- 'params': simulation constants (c, dt, dx)
"""

import os
import argparse
import numpy as np
from tqdm import tqdm
from phicnet.physics.wave_equation import generate_wave_trajectory


def generate_dataset_split(
    num_sequences: int,
    grid_size: int = 64,
    seq_len: int = 100,
    c: float = 2.0,
    dt: float = 0.005,
    seed: int = 42,
    split_name: str = "train"
):
    rng = np.random.default_rng(seed)
    u_all = np.zeros((num_sequences, seq_len, grid_size, grid_size), dtype=np.float32)
    s_all = np.zeros((num_sequences, seq_len, grid_size, grid_size), dtype=np.float32)

    # Distribute across source types: coupled_oscillators (60%), single_oscillator (25%), pulse (15%)
    source_choices = ["coupled_oscillators", "single_oscillator", "pulse"]
    source_weights = [0.6, 0.25, 0.15]

    print(f"Generating {split_name} split ({num_sequences} sequences of length {seq_len})...")
    for i in tqdm(range(num_sequences), desc=f"Generating {split_name}"):
        stype = rng.choice(source_choices, p=source_weights)
        u_traj, s_traj, _ = generate_wave_trajectory(
            grid_size=grid_size,
            num_steps=seq_len,
            c=c,
            dt=dt,
            source_type=stype,
            rng=rng
        )
        u_all[i] = u_traj
        s_all[i] = s_traj

    return u_all, s_all


def main():
    parser = argparse.ArgumentParser(description="Generate 2D Wave Propagation Data for PhICNet")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory to save generated datasets")
    parser.add_argument("--train_seqs", type=int, default=60, help="Number of training sequences")
    parser.add_argument("--val_seqs", type=int, default=15, help="Number of validation sequences")
    parser.add_argument("--test_seqs", type=int, default=15, help="Number of test sequences")
    parser.add_argument("--seq_len", type=int, default=100, help="Frames per sequence")
    parser.add_argument("--grid_size", type=int, default=64, help="Grid size H=W")
    parser.add_argument("--c", type=float, default=10.0, help="Wave propagation speed c")
    parser.add_argument("--dt", type=float, default=0.001, help="Time step dt")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    dx = 1.0 / (args.grid_size - 1)

    splits = [
        ("train", args.train_seqs, args.seed),
        ("val", args.val_seqs, args.seed + 100),
        ("test", args.test_seqs, args.seed + 200),
    ]

    for name, n_seq, seed in splits:
        u_data, s_data = generate_dataset_split(
            num_sequences=n_seq,
            grid_size=args.grid_size,
            seq_len=args.seq_len,
            c=args.c,
            dt=args.dt,
            seed=seed,
            split_name=name
        )
        filepath = os.path.join(args.data_dir, f"{name}.npz")
        np.savez_compressed(
            filepath,
            u=u_data,
            s=s_data,
            c=args.c,
            dt=args.dt,
            dx=dx,
            grid_size=args.grid_size
        )
        print(f"Saved {name} data to {filepath} (u: {u_data.shape}, s: {s_data.shape})")

    print("\nData generation complete!")


if __name__ == "__main__":
    main()
