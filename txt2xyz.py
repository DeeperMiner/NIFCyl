# -*- coding: utf-8 -*-
"""
Convert .txt point cloud (x y z) to .xyz
"""

import numpy as np

def txt_to_xyz(in_path, out_path):
    try:
        pts = np.loadtxt(in_path, dtype=np.float64)
    except ValueError:
        pts = np.loadtxt(in_path, dtype=np.float64, delimiter=',')
    pts = pts[:, :3]
    np.savetxt(out_path, pts, fmt="%.6f")

if __name__ == "__main__":
    in_file = " "
    out_file = " "
    txt_to_xyz(in_file, out_file)
    print(f"[OK] Saved to {out_file}")



