# catalog parsing patterns and elementary lattice geometry helpers

import re
import numpy as np

# regex patterns
ex_line_re = re.compile(r"Ex\s*=\s*([\d\.\-+Ee]+),\s*Ey\s*=\s*([\d\.\-+Ee]+),\s*Ez\s*=\s*([\d\.\-+Ee]+)")
g_line_re  = re.compile(r"Gyz\s*=\s*([\d\.\-+Ee]+),\s*Gxz\s*=\s*([\d\.\-+Ee]+),\s*Gxy\s*=\s*([\d\.\-+Ee]+)")
cell_header      = "Normalized unit cell parameters"
joint_pos_header = "Nodal positions:"
strut_header     = "Bar connectivities:"
small_re   = re.compile(r".*very small members.*")
overlap_re = re.compile(r".*overlapping or intersecting bars.*")
other_re   = re.compile(r"Other name\(s\):\s*(.*)")

# construct cell vectors from lengths and inter-vector angles in degrees
def lattice_matrix(a, b, c, alpha, beta, gamma):
    ar, br, gr = np.deg2rad([alpha, beta, gamma])
    v1 = np.array([a, 0, 0])
    v2 = np.array([b*np.cos(gr), b*np.sin(gr), 0])
    cx = c*np.cos(br)
    cy = c*(np.cos(ar) - np.cos(br)*np.cos(gr)) / np.sin(gr)
    cz = np.sqrt(max(0.0, c**2 - cx*cx - cy*cy))
    v3 = np.array([cx, cy, cz])
    return np.vstack([v1, v2, v3]).T

# detect coordinates that coincide at the requested decimal precision
def has_dup(pos, tol=6):
    seen = {tuple(round(c, tol) for c in p) for p in pos}
    return len(seen) < len(pos)

# orient a strut deterministically without changing its physical axis
def canonical_order(i, j, pos_cart, tol=1e-12):
    u, v = pos_cart[i], pos_cart[j]
    vec = v - u
    if vec[0] >  tol: return i, j
    if vec[0] < -tol: return j, i
    if vec[1] >  tol: return i, j
    if vec[1] < -tol: return j, i
    if vec[2] >= 0:   return i, j
    return j, i
