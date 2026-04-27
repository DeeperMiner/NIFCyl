import numpy as np

f1_dir = ''
f2_dir = ""

merged_dir = ''

file1 = np.loadtxt(f1_dir)
file2 = np.loadtxt(f2_dir)

a = np.array(file1)  # original coordinates, deformed coordinates, gt deformation
b = np.array(file2)[:, 3:6]  # ngf normals

c = np.hstack((a, b))

np.savetxt(merged_dir, c, fmt='%.6f')


