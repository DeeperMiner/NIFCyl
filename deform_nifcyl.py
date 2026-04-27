#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute deformation using NIFCyl.

Input TXT columns (1-indexed in the description, 0-indexed in code):
  1-3  [0:3]   = original point cloud coords  (x0, y0, z0) with reconstructed surface
  4-6  [3:6]   = deformed point cloud coords  (x1, y1, z1)
  7    [6]     = ground-truth deformation (kept unchanged in output)
  8-10 [7:10]  = normal fields derived from trained SDF (nx, ny, nz) 
"""

import math
import sys
from typing import Tuple
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

try:
    from scipy.spatial import cKDTree as KDTree
except Exception:
    KDTree = None


def load_txt_any(path: str) -> np.ndarray:
    try:
        arr = np.loadtxt(path, dtype=np.float64)
    except ValueError:
        arr = np.loadtxt(path, dtype=np.float64, delimiter=',')
    if arr.ndim != 2 or arr.shape[1] < 10:
        raise ValueError(f"Expect at least 10 columns, got shape {arr.shape}.")
    return arr


def normalize_rows(v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(v, axis=1)
    vn = np.zeros_like(v)
    mask = norms > 0
    vn[mask] = v[mask] / norms[mask, None]
    return vn, norms


def cylinder_candidates(tree: KDTree, center: np.ndarray, R: float, L: float) -> np.ndarray:
    R_sphere = math.sqrt(R * R + L * L)
    return tree.query_ball_point(center, r=R_sphere)


def inside_cylinder(points: np.ndarray, center: np.ndarray, axis_n: np.ndarray, R: float, L: float):
    v = points - center[None, :]
    s = v @ axis_n
    vv = np.einsum('ij,ij->i', v, v)
    rho2 = vv - s * s
    rho2 = np.maximum(rho2, 0.0)
    mask = (np.abs(s) <= L) & (rho2 <= R * R)
    return mask, s, rho2


def estimate_deformation(P0: np.ndarray, P1: np.ndarray, N: np.ndarray,
                         radius: float, half_length: float, min_neighbors: int,
                         use_zero_for_nan: bool) -> np.ndarray:
    npts = P0.shape[0]
    out = np.full(npts, np.nan, dtype=np.float64)

    if KDTree is None:
        tree0 = tree1 = None
    else:
        tree0 = KDTree(P0)
        tree1 = KDTree(P1)

    for i in tqdm(range(npts)):
        n = N[i]
        if not np.isfinite(n).all() or (n * n).sum() == 0.0:
            continue
        c = P0[i]
        if tree0 is None:
            idx0 = np.arange(npts)
            idx1 = np.arange(npts)
        else:
            idx0 = cylinder_candidates(tree0, c, radius, half_length)
            idx1 = cylinder_candidates(tree1, c, radius, half_length)

        if len(idx0) == 0 or len(idx1) == 0:
            continue

        m0, s0, _ = inside_cylinder(P0[idx0], c, n, radius, half_length)
        m1, s1, _ = inside_cylinder(P1[idx1], c, n, radius, half_length)

        s0_in = s0[m0]
        s1_in = s1[m1]

        if (s0_in.size >= min_neighbors) and (s1_in.size >= min_neighbors):
            out[i] = float(s1_in.mean() - s0_in.mean())

    if use_zero_for_nan:
        out = np.nan_to_num(out, nan=0.0)
    return out


if __name__ == '__main__':
    in_path = ""
    radius = 0.02
    half_length = 0.4  
    min_neighbors = 5
    use_zero_for_nan = False 
    precision = 6

    data = load_txt_any(in_path)
    P0 = data[:, 0:3].astype(np.float64)
    P1 = data[:, 3:6].astype(np.float64)
    N_raw = data[:, 7:10].astype(np.float64)

    N, norms = normalize_rows(N_raw)
    R = float(radius)
    L = 2.0 * R if (half_length is None) else float(half_length)

    est = estimate_deformation(P0, P1, N, R, L, min_neighbors, use_zero_for_nan)
    out = np.hstack([data, est.reshape(-1, 1)])
    ## r2, rmse and plot
    gt = np.array(out)[:, 6]
    d_ngf = np.array(out)[:, 10]

    mask = (gt != 0) & ~np.isnan(gt) & ~np.isnan(d_ngf)
    gt_m = gt[mask]
    d_ngf_m = d_ngf[mask]

    x = gt_m
    y = d_ngf_m

    r2 = r2_score(x, y)
    rmse = np.sqrt(mean_squared_error(x, y))

    print('r2', r2)
    print('rmse', rmse)
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, alpha=0.5, s=10, c="blue")
    plt.xlabel("Actual Deformation", fontsize=50)
    plt.ylabel("NIFCyl Deformation", fontsize=50)
    plt.title("NIFCyl vs Actual Deformation", fontsize=50)

    # Add y = x line
    # min_val = min(x.min(), y.min())
    min_val = min(0, 0)
    max_val = max(x.max(), y.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="y = x")

    # R2 and RMSE text
    plt.text(0.05, 0.95, f"R² = {r2:.3f}\nRMSE = {rmse:.3f}",
             transform=plt.gca().transAxes,
             fontsize=25, verticalalignment="top")

    plt.legend()
    plt.grid(False)
    plt.show()

    fmt = '%.{}f'.format(max(0, int(precision)))
