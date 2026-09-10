# dataset parsing, periodic reconstruction, feature construction, and caching

import math
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data, HeteroData

# catalog headers and numeric patterns
from .utils import (
    lattice_matrix, ex_line_re, g_line_re,
    cell_header, joint_pos_header, strut_header,
    small_re, overlap_re, other_re, has_dup, canonical_order
)
from .fundamental_representation import (
    AuthorGeometryRejectedError,
    FundamentalRepresentationError,
    build_grega_fundamental_representation,
    validate_author_node_distance,
)

# visualization data shared with viz.py
# viz.py imports these directly
structures: Dict[str, dict] = {}  # filled by _ensure_structures()
CENTRALITY_MAX_ITER = 5000
CENTRALITY_TOL = 1e-6
CYCLIC_TARGET_PERMUTATIONS = (
    (1, 2, 0, 4, 5, 3),
    (2, 0, 1, 5, 3, 4),
)

# populate module-level `structures` for viz.py once per process
# uses the same strict parser as training so ids match exactly
def _ensure_structures(catalog_path: Path):
    global structures
    if structures:
        return
    raw = _parse_catalog_eth_style(Path(catalog_path))
    # keep only fields that viz.py reads
    slim = {}
    for name, entry in raw.items():
        slim[name] = {
            "pos": entry.get("pos", []),
            "cell": entry.get("cell", []),
            "struts": entry.get("struts", []),
            # optional targets retained if present
            "targets": entry.get("targets", []),
        }
    structures = slim

# returns (pos_cart, pos_frac, struts_all, virtual_joint_ids)
# signature matches viz.py usage: _, _, struts_all, vj = strut_features(name, structures)
def strut_features(name: str, _structs) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]], set]:
    if not isinstance(_structs, dict) or name not in _structs:
        raise KeyError(f"structure '{name}' not found")

    entry = _structs[name]
    pos_frac = np.vstack(entry["pos"]).astype(np.float64)
    a, b, c, alpha, beta, gamma = entry["cell"]
    L = lattice_matrix(a, b, c, alpha, beta, gamma)
    pos_cart = pos_frac @ L.T

    raw_edges = [(int(i), int(j)) for (i, j) in entry["struts"]]
    struts_all = [canonical_order(i, j, pos_cart) for (i, j) in raw_edges]
    vj = identify_virtual_joints(pos_frac, np.asarray(struts_all, dtype=np.int64))

    return pos_cart.astype(np.float32), pos_frac.astype(np.float32), struts_all, vj


# identify catalog boundary nodes that terminate exactly one cut segment
def identify_virtual_joints(pos_frac: np.ndarray,
                            struts_0: np.ndarray,
                            tol: float = 1e-6) -> set:
    if pos_frac.size == 0 or struts_0.size == 0:
        return set()
    arr = np.asarray(struts_0, dtype=np.int64)
    deg = np.zeros(pos_frac.shape[0], dtype=np.int64)
    for i, j in arr:
        if 0 <= i < deg.shape[0]:
            deg[i] += 1
        if 0 <= j < deg.shape[0]:
            deg[j] += 1
    frac = np.mod(pos_frac, 1.0)
    boundary = np.any(np.isclose(frac, 0.0, atol=tol) | np.isclose(frac, 1.0, atol=tol), axis=1)
    return {int(idx) for idx in range(deg.shape[0]) if deg[idx] == 1 and boundary[idx]}


try:
    from .config import CFG, REL_DENSITY, STRUT_RADIUS_MM
except Exception:
    CFG = {}
    REL_DENSITY = 0.01
    STRUT_RADIUS_MM = 0.1

CATALOG_FPS_LENGTH_RELATIVE_TOLERANCE = 1e-3

# deterministic local seed

# seed preprocessing operations without coupling them to training seeds
def set_global_seed(seed: int):
    try:
        import random
        random.seed(seed)
    except Exception:
        pass
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
    except Exception:
        pass

# spherical harmonic features describing strut directions
# encode undirected strut axes with even real spherical harmonics
# unit_dirs has shape [number of struts, 3]
# lmax is the highest spherical harmonic degree included
def _real_sh_dir(unit_dirs: np.ndarray, lmax: int) -> np.ndarray:
    from e3nn.o3 import spherical_harmonics

    with torch.no_grad():
        xyz = torch.as_tensor(unit_dirs, dtype=torch.get_default_dtype(), device="cpu")
        if xyz.ndim != 2 or xyz.size(1) != 3:
            raise ValueError("unit_dirs must have shape [N, 3] for e3nn spherical harmonics")
        xyz = torch.nn.functional.normalize(xyz, dim=1)
        # a cylindrical strut is an undirected axis: u and -u are physically identical
        # odd-l harmonics change sign under that reversal, so only even orders are valid strut-node features
        ls = [l for l in range(2, int(lmax) + 1) if l % 2 == 0]
        if not ls:
            return np.zeros((int(xyz.size(0)), 0), dtype=np.float32)
        y = spherical_harmonics(ls, xyz, normalize=True, normalization="component")
        if torch.is_complex(y):
            y = y.real
        return y.to(dtype=torch.float32, device="cpu").numpy()

# catalog discovery
# locate the Lumpe and Stankovic (2021) catalog in supported project locations
def _find_catalog(project_root: Path) -> Path | None:
    candidates: List[Path] = []
    for rel in [
        "Unit_Cell_Catalog.txt",
        "unit_cell_catalog.txt",
        "data/Unit_Cell_Catalog.txt",
        "dataset/Unit_Cell_Catalog.txt",
        "assets/Unit_Cell_Catalog.txt",
    ]:
        candidates.append(project_root / rel)
    for p in project_root.rglob("*.txt"):
        if "unit_cell_catalog" in p.name.lower():
            candidates.append(p)
    for p in candidates:
        if p.is_file():
            return p
    return None

# parse the Lumpe and Stankovic (2021) catalog
# expected labels include:
#   - Name:
#   - Normalized unit cell parameters
#   - Nodal positions:
#   - Bar connectivities:
#   - Ex = …, Ey = …, Ez = …
#   - Gyz = …, Gxz = …, Gxy = …
# skips blocks flagged by 'very small members' or 'overlapping or intersecting bars'
# collect alternative structure names listed after "Other name(s):"
def _parse_catalog_eth_style(
    path: Path,
    *,
    return_stats: bool = False,
):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Lumpe and Stankovic (2021) catalog not found at: {path}"
        )

    with path.open("r", encoding="utf-8", errors="replace") as f:
        raw = f.read()

    alias_map: Dict[str, str] = {}
    structures_local: Dict[str, dict] = {}
    parser_counts = defaultdict(int)

    # the Lumpe and Stankovic (2021) catalog separates sections with 89 dashes
    for blk in raw.split("-" * 89):
        lines = [l for l in blk.splitlines() if l.strip()]
        if not lines:
            continue

        nm_line = next((l for l in lines if l.startswith("Name:")), None)
        if not nm_line:
            continue
        parser_counts["named_blocks"] += 1
        name = nm_line.split(":", 1)[1].strip()

        # skip catalog blocks explicitly flagged as invalid
        has_small_member_flag = any(small_re.match(l) for l in lines)
        has_overlap_flag = any(overlap_re.match(l) for l in lines)
        if has_small_member_flag:
            parser_counts["condition_hit_very_small_members"] += 1
        if has_overlap_flag:
            parser_counts[
                "condition_hit_overlapping_or_intersecting_bars"
            ] += 1
        if has_small_member_flag and has_overlap_flag:
            parser_counts["condition_hit_both_catalog_flags"] += 1

        # primary reasons are mutually exclusive so their sum equals the number of uniquely excluded structures
        if has_small_member_flag:
            parser_counts["flagged_very_small_members"] += 1
            continue
        if has_overlap_flag:
            parser_counts["flagged_overlapping_or_intersecting_bars"] += 1
            continue

        # alias handling
        alt = next((l for l in lines if l.startswith("Other name(s):")), None)
        aliases = (
            [value.strip() for value in other_re.match(alt).group(1).split(",") if value.strip()]
            if alt and other_re.match(alt)
            else []
        )
        if any(a in alias_map for a in [name] + aliases):
            parser_counts["duplicate_name_or_alias"] += 1
            continue
        for a in [name] + aliases:
            alias_map[a] = name

        # elastic stiffness targets
        m_ex = next((ex_line_re.search(l) for l in lines if ex_line_re.search(l)), None)
        if not m_ex:
            parser_counts["missing_normal_moduli"] += 1
            continue
        ex_val, ey_val, ez_val = (float(m_ex.group(i)) for i in (1, 2, 3))
        m_g = next((g_line_re.search(l) for l in lines if g_line_re.search(l)), None)
        if not m_g:
            parser_counts["missing_shear_moduli"] += 1
            continue
        gyz, gxz, gxy = (float(m_g.group(i)) for i in (1, 2, 3))

        # cell params line after header
        a = b = c = alpha = beta = gamma = None
        for i, l in enumerate(lines):
            if cell_header in l:
                try:
                    a, b, c, alpha, beta, gamma = [float(x) for x in lines[i + 1].split(",")[:6]]
                except Exception:
                    pass
                break
        if None in (a, b, c, alpha, beta, gamma):
            parser_counts["missing_or_malformed_cell_parameters"] += 1
            continue

        # nodes & bars
        pos, bars, mode = [], [], None
        for l in lines:
            if joint_pos_header in l:
                mode = "n"; continue
            if strut_header in l:
                mode = "b"; continue
            if mode == "n":
                pts = list(map(float, l.split()[:3]))
                if len(pts) == 3:
                    pos.append(tuple(pts))
            elif mode == "b":
                pr = l.split()
                if len(pr) == 2:
                    bars.append((int(pr[0]) - 1, int(pr[1]) - 1))

        if has_dup(pos):
            parser_counts["duplicate_joint_coordinates_1e-6"] += 1
            continue

        targets = np.asarray(
            [ex_val, ey_val, ez_val, gyz, gxz, gxy], dtype=np.float64
        )
        if not np.all(np.isfinite(targets)) or np.any(targets <= 0.0):
            parser_counts["nonpositive_or_nonfinite_targets"] += 1
            continue

        structures_local[name] = {
            "E": (ex_val, ey_val, ez_val),
            "G": (gyz, gxz, gxy),
            "pos": pos,
            "struts": bars,
            "cell": (a, b, c, alpha, beta, gamma),
            "targets": targets.tolist(),
        }

    # drop geometry-free items
    for nm in list(structures_local):
        if not structures_local[nm]["pos"]:
            parser_counts["no_joints"] += 1
            structures_local.pop(nm)
        elif not structures_local[nm]["struts"]:
            parser_counts["no_struts"] += 1
            structures_local.pop(nm)

    parser_counts["retained_after_parser"] = len(structures_local)
    stats = dict(parser_counts)
    if return_stats:
        return structures_local, stats
    return structures_local

# feature construction
# convert fractional catalog coordinates to the unscaled Cartesian cell
def _unscaled_cell_geometry(pos_frac: np.ndarray,
                            cell_params: Tuple[float, float, float, float, float, float]) -> Tuple[np.ndarray, np.ndarray]:
    a, b, c, alpha, beta, gamma = cell_params
    cell_matrix = lattice_matrix(a, b, c, alpha, beta, gamma).astype(np.float32)
    pos_cart = pos_frac @ cell_matrix.T
    return pos_cart.astype(np.float32, copy=False), cell_matrix


# return Lumpe and Stankovic (2021)'s per-cell bar length with face/edge sharing weights
def _catalog_effective_total_strut_length(
    pos_frac: np.ndarray,
    struts_raw: np.ndarray,
    cell_matrix: np.ndarray,
    boundary_tol: float = 1e-6,
) -> float:
    positions = np.asarray(pos_frac, dtype=np.float64).reshape(-1, 3)
    struts = np.asarray(struts_raw, dtype=np.int64).reshape(-1, 2)
    matrix = np.asarray(cell_matrix, dtype=np.float64).reshape(3, 3)
    if len(struts) == 0:
        return 0.0
    if np.min(struts) < 0 or np.max(struts) >= len(positions):
        raise ValueError(
            "Lumpe and Stankovic (2021) catalog strut endpoint is outside "
            "the node array"
        )

    total_length = 0.0
    for source, target in struts:
        p0 = positions[int(source)]
        p1 = positions[int(target)]
        sharing_weight = 1.0
        for axis in range(3):
            on_lower_face = (
                abs(float(p0[axis])) <= boundary_tol
                and abs(float(p1[axis])) <= boundary_tol
            )
            on_upper_face = (
                abs(float(p0[axis]) - 1.0) <= boundary_tol
                and abs(float(p1[axis]) - 1.0) <= boundary_tol
            )
            if on_lower_face or on_upper_face:
                sharing_weight *= 0.5
        vector = matrix @ (p1 - p0)
        total_length += sharing_weight * float(np.linalg.norm(vector))
    return float(total_length)


# record agreement between catalog and reconstructed per-cell strut length
def _attach_catalog_length_diagnostics(
    graph: Dict[str, object],
    catalog_total_length: float,
) -> float:
    vectors = np.asarray(graph["strut_vectors"], dtype=np.float64).reshape(-1, 3)
    fundamental_total_length = float(np.linalg.norm(vectors, axis=1).sum())
    relative_error = abs(fundamental_total_length - catalog_total_length) / max(
        abs(float(catalog_total_length)), 1e-15
    )
    graph["catalog_effective_total_length_unscaled"] = float(
        catalog_total_length
    )
    graph["fundamental_total_length_unscaled"] = float(fundamental_total_length)
    graph["catalog_fundamental_length_relative_error"] = float(relative_error)
    return float(relative_error)


# scale a graph to the radius and relative density used by Lumpe and Stankovic (2021)
def _scale_periodic_graph_to_density(graph: Dict[str, object],
                                     cell_matrix: np.ndarray,
                                     rel_density: float,
                                     strut_radius_mm: float) -> Tuple[Dict[str, object], np.ndarray]:
    cell_matrix = np.asarray(cell_matrix, dtype=np.float64).reshape(3, 3)
    vectors = np.asarray(graph["strut_vectors"], dtype=np.float64).reshape(-1, 3)
    s = 1.0
    if vectors.size and rel_density > 0.0 and strut_radius_mm > 0.0:
        total_length0 = float(np.linalg.norm(vectors, axis=1).sum())
        cell_volume0 = float(abs(np.linalg.det(cell_matrix)))
        if total_length0 > 0.0 and cell_volume0 > 0.0:
            s = math.sqrt(
                math.pi * float(strut_radius_mm) ** 2 * total_length0
                / (float(rel_density) * cell_volume0)
            )

    scaled = dict(graph)
    scaled["pos_cart"] = (
        np.asarray(graph["pos_cart"], dtype=np.float64) * s
    ).astype(np.float32)
    scaled["strut_vectors"] = (vectors * s).astype(np.float32)
    scaled["density_scale"] = float(s)
    cell_matrix_scaled = (cell_matrix * s).astype(np.float32)
    scaled["cell_matrix_scaled"] = cell_matrix_scaled
    scaled["relative_density"] = float(rel_density)
    scaled["strut_radius_mm"] = float(strut_radius_mm)
    return scaled, cell_matrix_scaled

# forward unit directions to the undirected spherical-harmonic encoder
def _sh_features_from_dirs(unit_dirs: np.ndarray, lmax: int) -> np.ndarray:
    return _real_sh_dir(unit_dirs, lmax=lmax)

# read one configuration flag with an explicit default
def _cfg_bool(name: str, default: bool = True) -> bool:
    return bool(CFG.get(name, default))


# raised when an exact centrality feature cannot be computed
class CentralityComputationError(RuntimeError):
    pass


# raised when eigenvector centrality reaches its iteration limit
class CentralityNonConvergenceError(CentralityComputationError):
    pass



# encode scaled cell size ratios and the three cell-angle cosines
def _cell_feature(cell_params: Tuple[float, float, float, float, float, float],
                  cell_matrix_scaled: np.ndarray | None = None) -> np.ndarray:
    if not _cfg_bool("INCLUDE_CELL_PARAMS", True):
        return np.zeros((1, 0), dtype=np.float32)
    a, b, c, alpha, beta, gamma = [float(x) for x in cell_params]
    if cell_matrix_scaled is not None:
        L = np.asarray(cell_matrix_scaled, dtype=np.float64).reshape(3, 3)
        lengths = np.linalg.norm(L, axis=0)
    else:
        lengths = np.asarray([a, b, c], dtype=np.float64)
    if not np.all(np.isfinite(lengths)) or np.any(lengths <= 0.0):
        raise ValueError("cell-vector lengths must be positive and finite")
    a_scaled, b_scaled, c_scaled = [float(x) for x in lengths]
    angle_cosines = np.cos(
        np.deg2rad(np.asarray([alpha, beta, gamma], dtype=np.float64))
    )
    features = np.asarray(
        [
            np.log(a_scaled),
            np.log(b_scaled / a_scaled),
            np.log(c_scaled / a_scaled),
            *angle_cosines.tolist(),
        ],
        dtype=np.float32,
    )
    return features.reshape(1, 6)

# compute multiplicity-aware degree, closeness, betweenness, and eigenvector centralities
def _centrality_features(
    G: nx.Graph,
    num_nodes: int,
    *,
    degree_weight: str | None = None,
    closeness_distance: str | None = None,
    betweenness_distance: str | None = None,
    eigenvector_weight: str | None = None,
) -> np.ndarray:
    G = G.copy()
    G.add_nodes_from(range(num_nodes))
    if num_nodes == 0 or G.number_of_edges() == 0:
        return np.zeros((num_nodes, 4), dtype=np.float32)

    try:
        degree_scale = 1.0 / float(num_nodes - 1) if num_nodes > 1 else 1.0
        dc_raw = {
            int(node): float(value) * degree_scale
            for node, value in G.degree(weight=degree_weight)
        }
    except Exception as exc:
        raise CentralityComputationError(
            f"weighted degree centrality failed: {exc or type(exc).__name__}"
        ) from exc
    dc = np.array(
        [dc_raw.get(i, 0.0) for i in range(num_nodes)], dtype=np.float32
    )[:, None]

    try:
        closeness_raw = nx.closeness_centrality(
            G,
            distance=closeness_distance,
            wf_improved=True,
        )
    except Exception as exc:
        raise CentralityComputationError(
            f"weighted closeness centrality failed: {exc or type(exc).__name__}"
        ) from exc
    closeness = np.array(
        [closeness_raw.get(i, 0.0) for i in range(num_nodes)],
        dtype=np.float32,
    )[:, None]

    try:
        bc_raw = nx.betweenness_centrality(
            G, normalized=True, weight=betweenness_distance
        )
    except Exception as exc:
        raise CentralityComputationError(
            f"exact betweenness centrality failed: {exc or type(exc).__name__}"
        ) from exc
    bc = np.array(
        [bc_raw.get(i, 0.0) for i in range(num_nodes)], dtype=np.float32
    )[:, None]

    try:
        ev_raw = nx.eigenvector_centrality(
            G,
            max_iter=CENTRALITY_MAX_ITER,
            tol=CENTRALITY_TOL,
            weight=eigenvector_weight,
        )
    except nx.PowerIterationFailedConvergence as exc:
        raise CentralityNonConvergenceError(
            "eigenvector centrality did not converge within "
            f"{CENTRALITY_MAX_ITER} iterations"
        ) from exc
    except Exception as exc:
        raise CentralityComputationError(
            f"eigenvector centrality failed: {exc or type(exc).__name__}"
        ) from exc
    ev = np.array(
        [ev_raw.get(i, 0.0) for i in range(num_nodes)], dtype=np.float32
    )[:, None]

    return np.hstack([dc, closeness, bc, ev]).astype(np.float32)

# project joint incidences to the periodic strut line multigraph
# each fundamental-graph self-loop contributes two incidences at its joint
# if two strut orbits meet through several endpoint incidences, their line-graph edge keeps that multiplicity
# two incidences of the same strut orbit create a line-graph self-loop
def _periodic_strut_adjacency_records(
    endpoints: np.ndarray | List[Tuple[int, int]],
) -> List[Tuple[int, int, int]]:
    endpoints_arr = np.asarray(endpoints, dtype=np.int64).reshape(-1, 2)
    by_joint: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for sid, (u_raw, v_raw) in enumerate(endpoints_arr):
        by_joint[int(u_raw)][int(sid)] += 1
        by_joint[int(v_raw)][int(sid)] += 1

    multiplicities: Dict[Tuple[int, int], int] = defaultdict(int)
    for counts in by_joint.values():
        strut_ids = sorted(counts)
        for ai, sid_a in enumerate(strut_ids):
            count_a = int(counts[sid_a])
            self_pairs = count_a * (count_a - 1) // 2
            if self_pairs:
                multiplicities[(sid_a, sid_a)] += self_pairs
            for sid_b in strut_ids[ai + 1:]:
                multiplicities[(sid_a, sid_b)] += count_a * int(counts[sid_b])

    return [
        (int(a), int(b), int(multiplicity))
        for (a, b), multiplicity in sorted(multiplicities.items())
        if int(multiplicity) > 0
    ]





# build the fundamental periodic structure using the Grega et al. (2025) procedure
# compare its total strut length with the effective per-cell catalog length
# reject failed reconstructions and length mismatches instead of using an alternative reconstruction
def _build_periodic_structure_graph(pos_frac: np.ndarray,
                                    pos_cart: np.ndarray,
                                    struts_raw: np.ndarray,
                                    cell_matrix: np.ndarray | None = None) -> Dict[str, object]:
    if cell_matrix is None:
        raise FundamentalRepresentationError(
            "Grega et al. (2025) fundamental periodic structure construction requires a "
            "cell matrix"
        )

    enforce_min_node_separation = _cfg_bool(
        "DATA_FILTER_USE_MIN_NODE_SEPARATION", True
    )
    if enforce_min_node_separation:
        validate_author_node_distance(
            pos_frac,
            stage="before sanitation",
        )
    struts_array = np.asarray(struts_raw, dtype=np.int64).reshape(-1, 2)
    catalog_total_length = _catalog_effective_total_strut_length(
        pos_frac, struts_array, cell_matrix
    )
    if catalog_total_length <= 0.0:
        raise FundamentalRepresentationError(
            "Lumpe and Stankovic (2021) catalog geometry has zero effective "
            "strut length"
        )

    try:
        graph = build_grega_fundamental_representation(
            pos_frac,
            struts_array,
            cell_matrix,
            enforce_min_node_separation=enforce_min_node_separation,
        )
    except AuthorGeometryRejectedError:
        raise
    except (
        FundamentalRepresentationError,
        AssertionError,
        IndexError,
        ValueError,
    ) as exc:
        raise FundamentalRepresentationError(
            "Grega et al. (2025) reconstruction failed and non-Grega et al. "
            "(2025) fallback is disabled: "
            f"{exc or type(exc).__name__}"
        ) from exc

    _attach_catalog_length_diagnostics(graph, catalog_total_length)
    graph["fundamental_representation_fallback_reason"] = ""
    graph["grega_attempted"] = True
    _validate_fundamental_representation(graph)
    return graph


# check that integer edge labels reproduce the compact and tiled topology
def _validate_fundamental_representation(graph: Dict[str, object], expected_raw_struts: int | None = None) -> None:
    endpoints = np.asarray(graph["strut_endpoints"], dtype=np.int64).reshape(-1, 2)
    shifts = np.asarray(graph["strut_shifts"], dtype=np.int64).reshape(-1, 3)
    frac_vectors = np.asarray(graph["strut_fractional_vectors"], dtype=np.float64).reshape(-1, 3)
    keys = np.asarray(graph["key_order"], dtype=np.float64).reshape(-1, 3)
    if not (len(endpoints) == len(shifts) == len(frac_vectors)):
        raise ValueError("periodic edge arrays have inconsistent lengths")
    if len(endpoints):
        expected = keys[endpoints[:, 1]] - keys[endpoints[:, 0]] + shifts
        if not np.allclose(frac_vectors, expected, atol=5e-6, rtol=0.0):
            raise ValueError("integer shifts do not reconstruct periodic strut vectors")
        labels = [
            (int(u), int(v), *[int(x) for x in shift])
            for (u, v), shift in zip(endpoints, shifts)
        ]
        if len(labels) != len(set(labels)):
            raise ValueError("duplicate labeled edges remain in fundamental periodic structure")
        if np.max(np.abs(shifts)) > 2:
            raise ValueError(
                "periodic edge shift exceeds the Lumpe and Stankovic (2021) "
                "catalog 3x3x3 cover"
            )
        cover_multiplicity = np.prod(3 - np.abs(shifts), axis=1)
        if np.any(cover_multiplicity <= 0):
            raise ValueError("periodic edge cannot be re-expanded in a 3x3x3 cover")

    expected_line_adjacency = _periodic_strut_adjacency_records(endpoints)
    stored_line_adjacency = [
        (int(a), int(b), int(multiplicity))
        for a, b, multiplicity in graph.get("strut_line_adjacency", [])
    ]
    if stored_line_adjacency != expected_line_adjacency:
        raise ValueError("periodic strut line-graph multiplicities are inconsistent")

    if expected_raw_struts is not None:
        sources = [
            int(source)
            for edge_sources in graph.get("strut_sources", [])
            for source in edge_sources
        ]
        if sorted(sources) != list(range(int(expected_raw_struts))):
            raise ValueError(
                "raw Lumpe and Stankovic (2021) catalog struts are not "
                "represented exactly once"
            )

# public graph builder for visualization and diagnostics
def build_periodic_structure_graph(name: str, _structs: Dict[str, dict] | None = None) -> Dict[str, object]:
    src = structures if _structs is None else _structs
    if not isinstance(src, dict) or name not in src:
        raise KeyError(f"structure '{name}' not found")
    entry = src[name]
    pos_frac = np.vstack(entry["pos"]).astype(np.float32)
    pos_cart, cell_matrix = _unscaled_cell_geometry(pos_frac, tuple(entry["cell"]))
    struts_raw = np.asarray(entry["struts"], dtype=np.int64)
    graph = _build_periodic_structure_graph(pos_frac, pos_cart, struts_raw, cell_matrix)
    graph, _ = _scale_periodic_graph_to_density(
        graph,
        cell_matrix,
        REL_DENSITY,
        STRUT_RADIUS_MM,
    )
    return graph

# return the multiplicity-weighted fundamental representation of the infinite line graph
def _periodic_strut_projection(graph: Dict[str, object]) -> nx.Graph:
    n = int(graph["num_struts"])
    G = nx.Graph()
    G.add_nodes_from(range(n))
    records = graph.get("strut_line_adjacency")
    if records is None:
        records = _periodic_strut_adjacency_records(graph["strut_endpoints"])
    for a_raw, b_raw, multiplicity_raw in records:
        a = int(a_raw)
        b = int(b_raw)
        multiplicity = float(multiplicity_raw)
        if multiplicity <= 0.0:
            continue
        # one line-graph self-loop represents two same-orbit neighbors in the periodic lift
        # networkx weighted degree counts a loop twice, but its adjacency operator uses the diagonal weight only once, so keep a separate spectral weight for eigenvector centrality
        spectral_multiplicity = 2.0 * multiplicity if a == b else multiplicity
        G.add_edge(
            a,
            b,
            multiplicity=multiplicity,
            distance=1.0 / multiplicity,
            spectral_multiplicity=spectral_multiplicity,
        )
    return G

# compute four centralities on the periodic multiplicity-weighted strut graph
def _compute_strut_line_features(graph: Dict[str, object]) -> np.ndarray:
    n = int(graph["num_struts"])
    G = _periodic_strut_projection(graph)
    features = _centrality_features(
        G,
        n,
        degree_weight="multiplicity",
        closeness_distance="distance",
        betweenness_distance="distance",
        eigenvector_weight="spectral_multiplicity",
    )
    return features

# project periodic strut orbits to a multiplicity-weighted joint graph
# distinct integer-shift edges between the same fundamental joints remain separate physical neighbors in the infinite lattice
# a periodic self-loop has two incidences at its fundamental joint, corresponding to the two opposite images.
def _periodic_joint_projection(graph: Dict[str, object]) -> nx.Graph:
    n = int(graph["num_joints"])
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for u_raw, v_raw in np.asarray(graph["strut_endpoints"], dtype=np.int64):
        u = int(u_raw)
        v = int(v_raw)
        strength_increment = 2.0 if u == v else 1.0
        degree_increment = 1.0
        if G.has_edge(u, v):
            G[u][v]["multiplicity"] += strength_increment
            G[u][v]["degree_multiplicity"] += degree_increment
        else:
            G.add_edge(
                u,
                v,
                multiplicity=strength_increment,
                degree_multiplicity=degree_increment,
            )

    for _, _, attrs in G.edges(data=True):
        multiplicity = float(attrs["multiplicity"])
        attrs["distance"] = 1.0 / multiplicity
    return G

# assemble joint-angle descriptors and periodic joint-graph centralities
def _compute_joint_features_periodic(graph: Dict[str, object], udir: np.ndarray) -> np.ndarray:
    n = int(graph["num_joints"])
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)

    parts = []
    boundary_cut_joint_ids = set(int(x) for x in graph.get("boundary_cut_joint_ids", set()))

    if _cfg_bool("JOINT_USE_ANGLES", True):
        incident_dirs = [[] for _ in range(n)]
        endpoints = np.asarray(graph["strut_endpoints"], dtype=np.int64)
        for sid, (u, v) in enumerate(endpoints):
            u = int(u)
            v = int(v)
            incident_dirs[int(u)].append(udir[sid])
            if v == u:
                incident_dirs[int(u)].append(-udir[sid])
            else:
                incident_dirs[int(v)].append(-udir[sid])

        ang_feats = np.zeros((n, 2), dtype=np.float32)
        for jid, dirs_list in enumerate(incident_dirs):
            if len(dirs_list) <= 1:
                continue
            dirs = np.asarray(dirs_list, dtype=np.float32)
            C = np.clip(dirs @ dirs.T, -1.0, 1.0)
            tri = np.triu_indices(C.shape[0], k=1)
            angs = np.arccos(C[tri])
            ang_feats[jid, 0] = float(angs.mean())
            ang_feats[jid, 1] = float(angs.std())
        parts.append(ang_feats)

    if _cfg_bool("JOINT_USE_CENTRALITY", True):
        G = _periodic_joint_projection(graph)
        cents = _centrality_features(
            G,
            n,
            degree_weight="degree_multiplicity",
            closeness_distance="distance",
            betweenness_distance="distance",
            eigenvector_weight="multiplicity",
        )
        if boundary_cut_joint_ids:
            for jid in boundary_cut_joint_ids:
                if 0 <= jid < n:
                    cents[jid, :] = 0.0
        parts.append(cents)

    if not parts:
        return np.zeros((n, 0), dtype=np.float32)
    return np.hstack(parts).astype(np.float32)

# assemble raw strut, joint, and unit cell feature matrices
def _build_feature_blocks(graph: Dict[str, object],
                          cell_params: Tuple[float, float, float, float, float, float],
                          cell_matrix_scaled: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    strut_vectors = np.asarray(graph["strut_vectors"], dtype=np.float32)
    if strut_vectors.size:
        lens = np.linalg.norm(strut_vectors, axis=1, keepdims=True) + 1e-9
        udir = strut_vectors / lens
    else:
        lens = np.zeros((0, 1), dtype=np.float32)
        udir = np.zeros((0, 3), dtype=np.float32)

    s_parts = []
    if _cfg_bool("STRUT_USE_LENGTH", True):
        s_parts.append(lens.astype(np.float32, copy=False))
    if _cfg_bool("STRUT_USE_SH", True):
        s_parts.append(_sh_features_from_dirs(udir, lmax=int(CFG["SH_LMAX"])))
    if _cfg_bool("STRUT_USE_CENTRALITY", True):
        s_parts.append(_compute_strut_line_features(graph))
    s_x = np.hstack(s_parts).astype(np.float32) if s_parts else np.zeros((int(graph["num_struts"]), 0), dtype=np.float32)

    j_x = _compute_joint_features_periodic(graph, udir)
    u_x = _cell_feature(cell_params, cell_matrix_scaled)
    return s_x, j_x, u_x

# record reconstruction removals for auditing rather than model input
def _periodic_removal_counts(entry: dict, graph: Dict[str, object]) -> np.ndarray:
    raw_num_joints = int(len(entry.get("pos", [])))
    num_virtual_removed = int(len(graph.get("virtual_joints", [])))
    num_periodic_removed = int(
        sum(max(0, len(group) - 1) for group in graph.get("groups", []))
    )
    num_boundary_degree1 = int(len(graph.get("boundary_degree1_joints", [])))
    return np.asarray(
        [[raw_num_joints, num_virtual_removed, num_periodic_removed, num_boundary_degree1]],
        dtype=np.float32,
    )

# exclude visualization-only boundary cut joints from learned pooling
def _joint_physical_mask(graph: Dict[str, object]) -> np.ndarray:
    n = int(graph["num_joints"])
    mask = np.ones((n,), dtype=bool)
    for jid in graph.get("boundary_cut_joint_ids", set()):
        jid = int(jid)
        if 0 <= jid < n:
            mask[jid] = False
    if n > 0 and not bool(mask.any()):
        mask[:] = True
    return mask

# encode joint and strut counts with log1p
def _graph_count_features(graph: Dict[str, object]) -> np.ndarray:
    physical_joint_count = int(np.count_nonzero(_joint_physical_mask(graph)))
    strut_count = int(graph["num_struts"])
    return np.log1p(
        np.asarray([[physical_joint_count, strut_count]], dtype=np.float32)
    ).astype(np.float32)

# record one reconstructed periodic representation for cache auditing
def _periodic_structure_fingerprint(
    graph: Dict[str, object],
    cell_params: Tuple[float, float, float, float, float, float],
) -> str:
    import hashlib

    digest = hashlib.sha256()
    arrays = (
        np.asarray(graph["strut_endpoints"], dtype=np.int64).reshape(-1, 2),
        np.asarray(graph["strut_fractional_vectors"], dtype=np.float64).reshape(-1, 3),
        np.asarray(cell_params, dtype=np.float64).reshape(1, 6),
    )
    for arr in arrays:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.round(arr, 6)
        arr = np.ascontiguousarray(arr)
        digest.update(str(arr.shape).encode("ascii"))
        digest.update(arr.dtype.str.encode("ascii"))
        digest.update(arr.tobytes())
    return digest.hexdigest()

# create aligned bipartite and homogeneous PyG representations
def _pack_hetero(entry: dict,
                 graph: Dict[str, object],
                 s_x: np.ndarray,
                 j_x: np.ndarray,
                 u_x: np.ndarray,
                 targets: np.ndarray) -> HeteroData:
    jN = int(graph["num_joints"])
    sN = int(graph["num_struts"])
    data = HeteroData()

    data["joint"].x = torch.from_numpy(j_x.astype(np.float32, copy=False))
    data["joint"].num_nodes = jN
    physical_mask = _joint_physical_mask(graph)
    data["joint"].physical_mask = torch.from_numpy(physical_mask)
    data["joint"].boundary_cut_mask = torch.from_numpy(~physical_mask)

    data["strut"].x = torch.from_numpy(s_x.astype(np.float32, copy=False))
    data["strut"].num_nodes = sN
    strut_vectors = np.asarray(graph["strut_vectors"], dtype=np.float32).reshape(-1, 3)
    strut_lengths = np.linalg.norm(strut_vectors, axis=1, keepdims=True)
    strut_unit_dirs = np.divide(
        strut_vectors,
        np.maximum(strut_lengths, 1e-12),
        out=np.zeros_like(strut_vectors),
    )
    data["strut"].augmentation_unit_dir = torch.from_numpy(strut_unit_dirs)

    pairs = []
    endpoints = np.asarray(graph["strut_endpoints"], dtype=np.int64)
    for sid, (u, v) in enumerate(endpoints):
        # every periodic strut has two endpoint incidences
        # for a fundamental representation self-loop both endpoints map to the same fundamental joint
        for jid in (int(u), int(v)):
            pairs.append((sid, jid))
    if pairs:
        arr = np.asarray(pairs, dtype=np.int64)
        edge_s_to_j = arr.T
        edge_j_to_s = arr[:, [1, 0]].T
    else:
        edge_s_to_j = np.zeros((2, 0), dtype=np.int64)
        edge_j_to_s = np.zeros((2, 0), dtype=np.int64)

    data[("strut", "incident", "joint")].edge_index = torch.from_numpy(edge_s_to_j)
    data[("joint", "has", "strut")].edge_index = torch.from_numpy(edge_j_to_s)

    # targets are log10 stiffness; reporting converts them back to MPa
    data["graph_y"] = torch.from_numpy(targets.reshape(1, 6).astype(np.float32))
    data.y = data["graph_y"]

    data.u = torch.from_numpy(u_x.astype(np.float32, copy=False))
    data.graph_counts = torch.from_numpy(_graph_count_features(graph))
    data.periodic_removal_counts = torch.from_numpy(_periodic_removal_counts(entry, graph))
    data.structure_fingerprint = _periodic_structure_fingerprint(graph, tuple(entry["cell"]))

    return data

# homogeneous graph: joints as nodes, struts as undirected edges.
# node features are the same periodic joint features used in the hetero path
def _pack_homog(entry: dict,
                graph: Dict[str, object],
                s_x: np.ndarray,
                j_x: np.ndarray,
                u_x: np.ndarray,
                targets: np.ndarray) -> Data:
    x = j_x.astype(np.float32, copy=False)
    directed_edges = []
    directed_features = []
    for sid, (u, v) in enumerate(np.asarray(graph["strut_endpoints"], dtype=np.int64)):
        u = int(u)
        v = int(v)
        directed_edges.append((u, v))
        directed_features.append(s_x[sid])
        # keep both endpoint incidences for fundamental graph self-loops as parallel self-loop message edges
        directed_edges.append((v, u))
        directed_features.append(s_x[sid])
    if directed_edges:
        e_idx = np.asarray(directed_edges, dtype=np.int64).T
        edge_attr = np.asarray(directed_features, dtype=np.float32)
    else:
        e_idx = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, int(s_x.shape[1])), dtype=np.float32)

    g = Data(
        x=torch.from_numpy(x.astype(np.float32, copy=False)),
        edge_index=torch.from_numpy(e_idx),
        y=torch.from_numpy(targets.reshape(1, 6).astype(np.float32)),
        num_nodes=int(graph["num_joints"]),
    )
    g.edge_attr = torch.from_numpy(edge_attr)
    g.u = torch.from_numpy(u_x.astype(np.float32, copy=False))
    g.graph_counts = torch.from_numpy(_graph_count_features(graph))
    g.periodic_removal_counts = torch.from_numpy(_periodic_removal_counts(entry, graph))
    g.structure_fingerprint = _periodic_structure_fingerprint(graph, tuple(entry["cell"]))
    physical_mask = _joint_physical_mask(graph)
    g.physical_mask = torch.from_numpy(physical_mask)
    g.boundary_cut_mask = torch.from_numpy(~physical_mask)
    return g

# record semantic widths needed for slicing and normalization
def _collect_dims_meta(s_x: np.ndarray, j_x: np.ndarray, u_x: np.ndarray) -> Dict[str, int]:
    length_dim = 1 if _cfg_bool("STRUT_USE_LENGTH", True) else 0
    sh_dim = (
        sum(2 * l + 1 for l in range(2, int(CFG["SH_LMAX"]) + 1) if l % 2 == 0)
        if _cfg_bool("STRUT_USE_SH", True)
        else 0
    )
    strut_centrality_dim = 4 if _cfg_bool("STRUT_USE_CENTRALITY", True) else 0
    joint_angles_dim = 2 if _cfg_bool("JOINT_USE_ANGLES", True) else 0
    joint_centrality_dim = 4 if _cfg_bool("JOINT_USE_CENTRALITY", True) else 0
    return {
        "s_total_dim": int(s_x.shape[1]) if s_x.size else 0,
        "j_total_dim": int(j_x.shape[1]) if j_x.size else 0,
        "s_geom_dim": int(length_dim + sh_dim),
        "s_length_dim": int(length_dim),
        "s_sh_dim": int(sh_dim),
        "s_centrality_dim": int(strut_centrality_dim),
        "j_angles_dim": int(joint_angles_dim),
        "j_centrality_dim": int(joint_centrality_dim),
        "u_dim": int(u_x.shape[1]) if u_x.size else 0,
    }

# combine per structure feature widths into one dataset schema
def _merge_dims_meta(metas: List[Dict[str, int]]) -> Dict[str, int]:
    out = {}
    for m in metas:
        for k, v in m.items():
            out[k] = max(out.get(k, 0), int(v))
    return out

# fit feature and log-target statistics on training structures only
def _compute_normalizers(hetero_list: List[HeteroData],
                         dims_meta: Dict[str, int] | None = None,
                         idxs: List[int] | None = None):
    out = {
        "joint": None,
        "strut": None,
        "u": None,
        "graph_counts": None,
        "y": None,
    }
    if len(hetero_list) == 0:
        return out

    def _has_node_type(h: HeteroData, ntype: str) -> bool:
        try:
            return ntype in h.node_types
        except Exception:
            return False

    def _stack(ntype: str):
        arrs = []
        source = range(len(hetero_list)) if idxs is None else idxs
        for i in source:
            if i < 0 or i >= len(hetero_list):
                continue
            h = hetero_list[i]
            if _has_node_type(h, ntype) and hasattr(h[ntype], "x"):
                arrs.append(h[ntype].x.float().numpy())
        if not arrs:
            return None
        return np.vstack(arrs)

    def _stack_graph_attr(attr: str):
        arrs = []
        source = range(len(hetero_list)) if idxs is None else idxs
        for i in source:
            if i < 0 or i >= len(hetero_list):
                continue
            value = getattr(hetero_list[i], attr, None)
            if value is not None and value.numel() > 0:
                arrs.append(value.float().numpy())
        if not arrs:
            return None
        return np.vstack(arrs)

    def _stats(arr: np.ndarray):
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(arr, axis=0)
            std = np.nanstd(arr, axis=0)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        std = np.where(np.isfinite(std), std, 1.0)
        std = np.where(std < 1e-8, 1.0, std)
        return {
            "mean": mean.astype(np.float32).tolist(),
            "std": std.astype(np.float32).tolist(),
            "missing": np.sum(~np.isfinite(arr), axis=0).astype(np.int64).tolist(),
        }

    jX = _stack("joint")
    sX = _stack("strut")
    uX = _stack_graph_attr("u")
    graph_counts = _stack_graph_attr("graph_counts")
    source = range(len(hetero_list)) if idxs is None else idxs
    Ys = np.vstack([
        hetero_list[i]["graph_y"].float().numpy()
        for i in source
        if 0 <= i < len(hetero_list)
    ])

    if jX is not None and jX.size:
        out["joint"] = _stats(jX)
    if sX is not None and sX.size:
        strut_stats = _stats(sX)
        normalize_mask = np.ones(int(sX.shape[1]), dtype=bool)
        if _cfg_bool("STRUT_USE_SH", True):
            sh_start = 1 if _cfg_bool("STRUT_USE_LENGTH", True) else 0
            sh_width = sum(
                2 * l + 1
                for l in range(2, int(CFG["SH_LMAX"]) + 1)
                if l % 2 == 0
            )
            sh_stop = sh_start + sh_width
            if sh_stop > int(sX.shape[1]):
                raise ValueError(
                    "configured spherical-harmonic feature width exceeds the "
                    "constructed strut feature width"
                )
            normalize_mask[sh_start:sh_stop] = False
        strut_stats["normalize_mask"] = normalize_mask.tolist()
        out["strut"] = strut_stats
    if uX is not None and uX.size:
        out["u"] = _stats(uX)
    if graph_counts is not None and graph_counts.size:
        out["graph_counts"] = _stats(graph_counts)
    if Ys.size:
        out["y"] = _stats(Ys)
    return out

# standardize only feature columns selected by the stored mask
def _normalize_tensor(x: torch.Tensor, stats: dict | None) -> torch.Tensor:
    if stats is None or x is None or x.numel() == 0:
        return x
    mean = torch.as_tensor(stats["mean"], dtype=x.dtype, device=x.device)
    std = torch.as_tensor(stats["std"], dtype=x.dtype, device=x.device)
    normalize_mask = torch.as_tensor(
        stats.get("normalize_mask", [True] * int(mean.numel())),
        dtype=torch.bool,
        device=x.device,
    )
    if normalize_mask.numel() != mean.numel():
        raise ValueError("feature normalization mask has the wrong width")
    mean = torch.where(normalize_mask, mean, torch.zeros_like(mean))
    std = torch.where(normalize_mask, std, torch.ones_like(std))
    return torch.nan_to_num((x - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)

# apply shared training statistics to aligned graph representations
def _apply_feature_normalizers(hetero_items: List[HeteroData],
                               homog_items: List[Data],
                               normalizers: dict) -> None:
    j_stats = normalizers.get("joint")
    s_stats = normalizers.get("strut")
    u_stats = normalizers.get("u")
    graph_count_stats = normalizers.get("graph_counts")
    y_stats = normalizers.get("y")
    def _normalize_target(target: torch.Tensor) -> torch.Tensor:
        if y_stats is None:
            raise ValueError("training-set log10 target statistics are required")
        return _normalize_tensor(target.float(), y_stats)

    def _has_node_type(h: HeteroData, ntype: str) -> bool:
        try:
            return ntype in h.node_types
        except Exception:
            return False
    for h in hetero_items:
        if _has_node_type(h, "joint") and hasattr(h["joint"], "x"):
            h["joint"].x = _normalize_tensor(h["joint"].x.float(), j_stats)
        if _has_node_type(h, "strut") and hasattr(h["strut"], "x"):
            h["strut"].x = _normalize_tensor(h["strut"].x.float(), s_stats)
        if hasattr(h, "u"):
            h.u = _normalize_tensor(h.u.float(), u_stats)
        if hasattr(h, "graph_counts"):
            h.graph_counts = _normalize_tensor(
                h.graph_counts.float(), graph_count_stats
            )
        if getattr(h, "graph_y", None) is not None:
            target = _normalize_target(h["graph_y"])
            h["graph_y"] = target
            h.y = target
    for g in homog_items:
        if hasattr(g, "x"):
            g.x = _normalize_tensor(g.x.float(), j_stats)
        if hasattr(g, "edge_attr"):
            g.edge_attr = _normalize_tensor(g.edge_attr.float(), s_stats)
        if hasattr(g, "u"):
            g.u = _normalize_tensor(g.u.float(), u_stats)
        if hasattr(g, "graph_counts"):
            g.graph_counts = _normalize_tensor(
                g.graph_counts.float(), graph_count_stats
            )
        if getattr(g, "y", None) is not None:
            g.y = _normalize_target(g.y)

# create duplicate safe splits and normalize every representation equally
def _set_splits_and_normalize(hetero_items: List[HeteroData],
                              homog_items: List[Data],
                              normalizers: dict | None,
                              names: List[str] | None = None):
    trf = float(CFG.get("TRAIN_FRAC", 0.8))
    vaf = float(CFG.get("VAL_FRAC", 0.1))
    idx_train, idx_val, idx_test, duplicate_groups = _make_grouped_splits(
        hetero_items, int(CFG["SPLIT_SEED"]), trf, vaf
    )
    out = _set_explicit_splits_and_normalize(
        hetero_items,
        homog_items,
        normalizers,
        idx_train,
        idx_val,
        idx_test,
        names,
    )
    out["split_strategy"] = "attributed_incidence_isomorphism_grouped"
    out["duplicate_input_groups"] = int(duplicate_groups)
    return out

# fit statistics on explicit training indices and record all split IDs
def _set_explicit_splits_and_normalize(
    hetero_items: List[HeteroData],
    homog_items: List[Data],
    normalizers: dict | None,
    idx_train: List[int],
    idx_val: List[int],
    idx_test: List[int],
    names: List[str] | None = None,
):
    split_lists = [list(map(int, values)) for values in (idx_train, idx_val, idx_test)]
    split_sets = [set(values) for values in split_lists]
    if (split_sets[0] & split_sets[1]) or (split_sets[0] & split_sets[2]) or (split_sets[1] & split_sets[2]):
        raise ValueError("train, validation, and test indices must be disjoint")
    if any(i < 0 or i >= len(hetero_items) for values in split_lists for i in values):
        raise IndexError("split index is outside the dataset")
    if not split_lists[0]:
        raise ValueError("training split cannot be empty")

    out = dict(normalizers or {})
    if names is not None:
        out["names"] = names
    out["idx_train"], out["idx_val"], out["idx_test"] = split_lists

    train_stats = _compute_normalizers(hetero_items, idxs=split_lists[0])
    cell_raw_train_stats = train_stats.get("u")
    train_stats["u"] = None
    for key, value in train_stats.items():
        out[key] = value
    out["cell_raw_train_stats"] = cell_raw_train_stats
    out["standardize_log_targets"] = True
    out["feature_normalization"] = (
        "joint_non_sh_strut_and_graph_count_train_split_zscore_"
        "sh_and_cell_features_unscaled_log_target_zscore"
    )
    _apply_feature_normalizers(hetero_items, homog_items, out)
    return out

# convert a floating point row to a stable node label
def _rounded_row(values, decimals: int) -> tuple:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = np.round(arr, int(decimals))
    arr[arr == 0.0] = 0.0
    return tuple(float(value) for value in arr)

# represent model inputs as an attributed incidence graph
def _input_incidence_graph(item: HeteroData, decimals: int = 6) -> nx.Graph:
    graph = nx.Graph()
    joint_x = item["joint"].x.detach().cpu().numpy()
    strut_x = item["strut"].x.detach().cpu().numpy()
    physical_mask = getattr(item["joint"], "physical_mask", None)
    if physical_mask is None:
        physical = np.ones(joint_x.shape[0], dtype=bool)
    else:
        physical = physical_mask.detach().cpu().numpy().astype(bool, copy=False)

    for joint_id, row in enumerate(joint_x):
        signature = ("joint", bool(physical[joint_id]), _rounded_row(row, decimals))
        graph.add_node(
            ("joint", int(joint_id)),
            signature=signature,
            wl_label=repr(signature),
        )
    for strut_id, row in enumerate(strut_x):
        signature = ("strut", _rounded_row(row, decimals))
        graph.add_node(
            ("strut", int(strut_id)),
            signature=signature,
            wl_label=repr(signature),
        )

    edge_index = item[("strut", "incident", "joint")].edge_index
    incidences = edge_index.detach().cpu().numpy().T
    for strut_id, joint_id in incidences:
        edge = (("strut", int(strut_id)), ("joint", int(joint_id)))
        if graph.has_edge(*edge):
            graph.edges[edge]["multiplicity"] += 1
        else:
            graph.add_edge(*edge, multiplicity=1)
    for _, _, attributes in graph.edges(data=True):
        attributes["wl_multiplicity"] = str(attributes["multiplicity"])
    return graph

# form a permutation-invariant candidate key before exact isomorphism checks
def _input_fingerprint(item: HeteroData, decimals: int = 6) -> str:
    import hashlib

    graph = _input_incidence_graph(item, decimals=decimals)
    wl_hash = nx.weisfeiler_lehman_graph_hash(
        graph,
        edge_attr="wl_multiplicity",
        node_attr="wl_label",
        iterations=10,
    )
    digest = hashlib.sha256()
    digest.update(wl_hash.encode("ascii"))
    for value in (
        getattr(item, "u", torch.zeros((1, 0))),
        getattr(item, "graph_counts", torch.zeros((1, 0))),
    ):
        arr = np.round(
            value.detach().cpu().numpy().astype(np.float64), int(decimals)
        )
        arr[arr == 0.0] = 0.0
        arr = np.ascontiguousarray(arr)
        digest.update(str(arr.shape).encode("ascii"))
        digest.update(arr.tobytes())
    return digest.hexdigest()

# confirm candidate duplicates by exact attributed graph isomorphism
def _input_isomorphic(first: HeteroData, second: HeteroData, decimals: int = 6) -> bool:
    for attribute in ("u", "graph_counts"):
        first_value = getattr(first, attribute, torch.zeros((1, 0)))
        second_value = getattr(second, attribute, torch.zeros((1, 0)))
        first_array = np.round(
            first_value.detach().cpu().numpy().astype(np.float64), int(decimals)
        )
        second_array = np.round(
            second_value.detach().cpu().numpy().astype(np.float64), int(decimals)
        )
        if first_array.shape != second_array.shape or not np.array_equal(
            first_array, second_array
        ):
            return False

    first_graph = _input_incidence_graph(first, decimals=decimals)
    second_graph = _input_incidence_graph(second, decimals=decimals)
    node_match = nx.algorithms.isomorphism.categorical_node_match(
        "signature", None
    )
    edge_match = nx.algorithms.isomorphism.categorical_edge_match(
        "multiplicity", None
    )
    return nx.is_isomorphic(
        first_graph,
        second_graph,
        node_match=node_match,
        edge_match=edge_match,
    )

# split without allowing identical model inputs to cross partitions
def _make_grouped_splits(
    items: List[HeteroData],
    seed: int,
    train_frac: float,
    val_frac: float,
) -> Tuple[List[int], List[int], List[int], int]:
    n_items = len(items)
    if n_items == 0:
        return [], [], [], 0
    if n_items < 3:
        return list(range(n_items)), [], [], 0

    groups_by_hash: Dict[str, List[int]] = defaultdict(list)
    for idx, item in enumerate(items):
        groups_by_hash[_input_fingerprint(item)].append(int(idx))
    groups: List[List[int]] = []
    for candidate_indices in groups_by_hash.values():
        exact_groups: List[List[int]] = []
        for idx in candidate_indices:
            for group in exact_groups:
                if _input_isomorphic(items[idx], items[group[0]]):
                    group.append(idx)
                    break
            else:
                exact_groups.append([idx])
        groups.extend(exact_groups)
    duplicate_groups = sum(1 for group in groups if len(group) > 1)

    rng = np.random.default_rng(int(seed))
    rng.shuffle(groups)
    n_train = min(max(1, int(round(train_frac * n_items))), n_items - 2)
    n_val = min(max(1, int(round(val_frac * n_items))), n_items - n_train - 1)
    targets = np.asarray([n_train, n_val, n_items - n_train - n_val], dtype=np.float64)
    buckets: List[List[int]] = [[], [], []]

    for group in groups:
        counts = np.asarray([len(bucket) for bucket in buckets], dtype=np.float64)
        relative_deficit = (targets - counts) / np.maximum(targets, 1.0)
        # random tie noise avoids a systematic preference for the training set
        choice = int(np.argmax(relative_deficit + rng.uniform(0.0, 1e-12, size=3)))
        buckets[choice].extend(group)

    for bucket in buckets:
        bucket.sort()
    return buckets[0], buckets[1], buckets[2], int(duplicate_groups)

# dataset cache
# create the reusable dataset-cache directory
def _cache_dir(project_root: Path) -> Path:
    d = project_root / "dataset_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d

# hash every setting that changes cached graph construction or features
def _cache_key(catalog_path: Path) -> str:
    import hashlib as _hashlib
    import e3nn as _e3nn
    sh_backend = f"e3nn-{getattr(_e3nn, '__version__', 'unknown')}"
    try:
        cat_stat = Path(catalog_path).stat()
        cat_mtime = float(cat_stat.st_mtime)
    except Exception:
        cat_mtime = 0.0

    # hash effective filter behavior only
    # a threshold cannot change the built dataset while its corresponding switch is disabled
    use_max_joints = _cfg_bool("DATA_FILTER_USE_MAX_JOINTS", True)
    use_max_struts = _cfg_bool("DATA_FILTER_USE_MAX_STRUTS", True)
    use_max_r_over_l = _cfg_bool(
        "DATA_FILTER_USE_MAX_RADIUS_LENGTH_RATIO", True
    )

    key_obj = {
        "catalog_abspath": str(Path(catalog_path).resolve()),
        "catalog_mtime": cat_mtime,
        "rel_density": REL_DENSITY,
        "strut_radius": STRUT_RADIUS_MM,
        "sh_lmax": int(CFG["SH_LMAX"]),
        "sh_backend": sh_backend,
        "dataset_filter": str(CFG["DATASET_FILTER"]).strip().lower(),
        "data_filter_use_max_joints": use_max_joints,
        "data_filter_max_joints": (
            _optional_positive_int("DATA_FILTER_MAX_JOINTS")
            if use_max_joints else None
        ),
        "data_filter_use_max_struts": use_max_struts,
        "data_filter_max_struts": (
            _optional_positive_int("DATA_FILTER_MAX_STRUTS")
            if use_max_struts else None
        ),
        "data_filter_use_max_radius_length_ratio": use_max_r_over_l,
        "data_filter_max_radius_length_ratio": (
            _optional_positive_float("DATA_FILTER_MAX_RADIUS_LENGTH_RATIO")
            if use_max_r_over_l else None
        ),
        "data_filter_use_min_node_separation": _cfg_bool(
            "DATA_FILTER_USE_MIN_NODE_SEPARATION", True
        ),
        "catalog_fps_length_relative_tolerance":
            CATALOG_FPS_LENGTH_RELATIVE_TOLERANCE,
        "strut_use_sh": bool(CFG.get("STRUT_USE_SH", True)),
        "strut_use_length": bool(CFG.get("STRUT_USE_LENGTH", True)),
        "strut_use_centrality": bool(CFG.get("STRUT_USE_CENTRALITY", True)),
        "joint_use_angles": bool(CFG.get("JOINT_USE_ANGLES", True)),
        "joint_use_centrality": bool(CFG.get("JOINT_USE_CENTRALITY", True)),
        "centrality_betweenness": "exact_all_retained_graphs",
        "centrality_feature_schema":
            "multiplicity_weighted_degree_closeness_betweenness_eigenvector_v1",
        "centrality_failure_policy": "discard_structure",
        "centrality_max_iter": CENTRALITY_MAX_ITER,
        "centrality_tol": CENTRALITY_TOL,
        "include_cell_params": bool(CFG.get("INCLUDE_CELL_PARAMS", True)),
        "graph_count_encoding": "log1p_physical_joint_count_log1p_strut_count",
        "cell_feature_encoding":
            "log_a_log_b_over_a_log_c_over_a_cos_alpha_cos_beta_cos_gamma",
        "feature_normalization":
            "joint_non_sh_strut_and_graph_count_train_split_zscore_"
            "sh_and_cell_features_unscaled_log_target_zscore",
        "code_rev": "v64_remove_strut_self_loop_indicator",
    }
    s = json.dumps(key_obj, sort_keys=True)
    return _hashlib.sha1(s.encode("utf-8")).hexdigest()

# return a positive floating point setting or disable the threshold
def _optional_positive_float(name: str):
    value = CFG.get(name, None)
    if value is None or value == "":
        return None
    value = float(value)
    return value if value > 0.0 else None

# return a positive integer setting or disable the threshold
def _optional_positive_int(name: str):
    value = CFG.get(name, None)
    if value is None or value == "":
        return None
    value = int(value)
    return value if value > 0 else None

# map reconstruction exceptions to stable audit categories
def _classify_periodic_failure(error: Exception) -> str:
    message = str(error).lower()
    if "0.05 cutoff before sanitation" in message:
        return "node_distance_below_0.05_before_sanitation"
    if "0.05 cutoff after sanitation" in message:
        return "node_distance_below_0.05_after_sanitation"
    if "zero effective strut length" in message:
        return "zero_effective_catalog_strut_length"
    if "windowing/fundamental tracing failed" in message:
        return "window_or_trace_failure_including_0.005_edge_rule"
    if "sanitation" in message:
        return "grega_sanitation_failure"
    if "periodic partner" in message:
        return "periodic_partner_failure"
    if (
        "raw lumpe and stankovic (2021) catalog struts are not represented "
        "exactly once"
    ) in message:
        return "raw_strut_representation_validation_failure"
    if "periodic edge" in message or "integer shifts" in message:
        return "periodic_edge_or_shift_validation_failure"
    return "other_periodic_reconstruction_or_scaling_failure"


_PARSER_FILTER_EXPLANATIONS = {
    "flagged_very_small_members":
        "Lumpe and Stankovic (2021) catalog entry contains very small members",
    "flagged_overlapping_or_intersecting_bars":
        "Lumpe and Stankovic (2021) catalog entry contains overlapping/intersecting bars",
    "duplicate_name_or_alias":
        "name or alias was assigned to an earlier Lumpe and Stankovic (2021) catalog entry",
    "missing_normal_moduli":
        "Ex, Ey, and Ez could not be read",
    "missing_shear_moduli":
        "Gyz, Gxz, and Gxy could not be read",
    "missing_or_malformed_cell_parameters":
        "six normalized unit-cell parameters could not be read",
    "duplicate_joint_coordinates_1e-6":
        "two Lumpe and Stankovic (2021) catalog joints coincide after rounding to six decimals",
    "nonpositive_or_nonfinite_targets":
        "at least one stiffness target is nonpositive or nonfinite",
    "no_joints":
        "no nodal positions were read",
    "no_struts":
        "no bar connectivities were read",
}


_PERIODIC_FILTER_EXPLANATIONS = {
    "malformed_strut_array":
        "bar connectivity does not have two endpoints",
    "node_distance_below_0.05_before_sanitation":
        "minimum reduced-coordinate node distance is below 0.05 before repair",
    "node_distance_below_0.05_after_sanitation":
        "minimum reduced-coordinate node distance is below 0.05 after repair",
    "zero_effective_catalog_strut_length":
        "Lumpe and Stankovic (2021) catalog geometry has zero total effective strut length",
    "window_or_trace_failure_including_0.005_edge_rule":
        "no valid Grega et al. (2025) window/trace was obtained; valid windows require edges > 0.005",
    "grega_sanitation_failure":
        "node-on-strut/intersection sanitation failed or remained unresolved",
    "periodic_partner_failure":
        "periodic face partners could not be assigned consistently",
    "raw_strut_representation_validation_failure":
        "raw struts were not represented exactly once by the fundamental representation",
    "periodic_edge_or_shift_validation_failure":
        "periodic endpoints, integer shifts, or cover validation failed",
    "other_periodic_reconstruction_or_scaling_failure":
        "another fundamental representation construction, validation, or physical scaling check failed",
}




# print catalog parsing exclusions in reproducible categories
def _print_catalog_preprocessing_counts(
    parser_counts: Dict[str, int],
) -> None:
    named = int(parser_counts.get("named_blocks", 0))
    retained_parser = int(parser_counts.get("retained_after_parser", 0))
    both_flags = int(
        parser_counts.get("condition_hit_both_catalog_flags", 0)
    )
    print(
        f"[data] read {named} structures from the "
        "Lumpe and Stankovic (2021) catalog"
    )
    print(
        "[data] discarded "
        f"{int(parser_counts.get('flagged_very_small_members', 0))} "
        "structures because the Lumpe and Stankovic (2021) catalog flagged "
        "very small members"
        + (
            f" ({both_flags} also had the overlapping-bars flag)"
            if both_flags else ""
        )
    )
    print(
        "[data] discarded "
        f"{int(parser_counts.get('flagged_overlapping_or_intersecting_bars', 0))} "
        "additional structures because the Lumpe and Stankovic (2021) catalog "
        "flagged overlapping or intersecting bars"
    )
    print(
        "[data] discarded "
        f"{int(parser_counts.get('duplicate_name_or_alias', 0))} "
        "structures because their name or alias duplicated an earlier entry"
    )
    for reason in (
        "missing_normal_moduli",
        "missing_shear_moduli",
        "missing_or_malformed_cell_parameters",
        "duplicate_joint_coordinates_1e-6",
        "nonpositive_or_nonfinite_targets",
        "no_joints",
        "no_struts",
    ):
        count = int(parser_counts.get(reason, 0))
        if count:
            print(
                f"[data] discarded {count} structures because "
                f"{_PARSER_FILTER_EXPLANATIONS[reason]}"
            )
    print(
        f"[data] {retained_parser} structures passed Lumpe and Stankovic "
        "(2021) catalog preprocessing"
    )


# print reconstruction and fixed quality filter outcomes
def _print_filter_audit(audit: Dict[str, object], *, cached: bool = False) -> None:
    parser_counts = dict(audit.get("parser_counts", {}))
    periodic_counts = dict(audit.get("periodic_rejection_counts", {}))
    quality_counts = dict(audit.get("quality_filter_counts", {}))
    sanitation_counts = dict(audit.get("sanitation_action_counts", {}))
    centrality_counts = dict(audit.get("centrality_rejection_counts", {}))

    if cached:
        _print_catalog_preprocessing_counts(parser_counts)

    if _cfg_bool("DATA_FILTER_USE_MIN_NODE_SEPARATION", True):
        print(
            "[data] discarded "
            f"{int(periodic_counts.get('node_distance_below_0.05_before_sanitation', 0))} "
            "structures because Grega et al. (2025) requires minimum node "
            "separation >= 0.05 before sanitation"
        )
        print(
            "[data] discarded "
            f"{int(periodic_counts.get('node_distance_below_0.05_after_sanitation', 0))} "
            "structures because Grega et al. (2025) requires minimum node "
            "separation >= 0.05 after sanitation"
        )
    else:
        print(
            "[data] Grega et al. (2025) minimum node-separation checks disabled"
        )
    print(
        "[data] discarded "
        f"{int(periodic_counts.get('window_or_trace_failure_including_0.005_edge_rule', 0))} "
        "structures because no Grega et al. (2025) periodic window with "
        "cropped edges > 0.005 was found"
    )
    standard_periodic_reasons = {
        "node_distance_below_0.05_before_sanitation",
        "node_distance_below_0.05_after_sanitation",
        "window_or_trace_failure_including_0.005_edge_rule",
    }
    for reason, count_raw in periodic_counts.items():
        count = int(count_raw)
        if count and reason not in standard_periodic_reasons:
            explanation = _PERIODIC_FILTER_EXPLANATIONS.get(
                reason, reason.replace("_", " ")
            )
            print(
                f"[data] discarded {count} structures because {explanation}"
            )

    nodes_split = int(
        sanitation_counts.get("structures_with_nodes_on_struts_split", 0)
    )
    intersections_split = int(
        sanitation_counts.get("structures_with_intersections_split", 0)
    )
    if nodes_split or intersections_split:
        print(
            f"[data] Grega et al. (2025) sanitation repaired {nodes_split} "
            f"structures with nodes on struts and {intersections_split} with "
            "strut intersections"
        )

    print(
        "[data] discarded "
        f"{int(quality_counts.get('catalog_fps_length_mismatch', 0))} "
        "structures because the catalog and fundamental periodic structure "
        "total strut lengths differ by at least 0.1%"
    )

    optional_reports = (
        (
            _cfg_bool("DATA_FILTER_USE_MAX_JOINTS", True),
            "too_many_joints",
            "Grega et al. (2025) limits fundamental joints to "
            f"{CFG.get('DATA_FILTER_MAX_JOINTS')}",
        ),
        (
            _cfg_bool("DATA_FILTER_USE_MAX_STRUTS", True),
            "too_many_struts",
            "Grega et al. (2025) limits fundamental struts to "
            f"{CFG.get('DATA_FILTER_MAX_STRUTS')}",
        ),
        (
            _cfg_bool("DATA_FILTER_USE_MAX_RADIUS_LENGTH_RATIO", True),
            "not_slender",
            "the Grega et al. (2025) slender-strut criterion requires r/L "
            f"< {CFG.get('DATA_FILTER_MAX_RADIUS_LENGTH_RATIO')} "
            "(using this project's scaled lengths)",
        ),
    )
    for enabled, reason, explanation in optional_reports:
        if enabled:
            print(
                f"[data] discarded {int(quality_counts.get(reason, 0))} "
                f"structures because {explanation}"
            )
    print(
        "[data] discarded "
        f"{int(centrality_counts.get('eigenvector_nonconvergence', 0))} "
        "structures because exact eigenvector centrality did not converge "
        f"within {CENTRALITY_MAX_ITER} iterations"
    )
    other_centrality_failures = int(
        centrality_counts.get("other_exact_centrality_failure", 0)
    )
    if other_centrality_failures:
        print(
            f"[data] discarded {other_centrality_failures} structures because "
            "another exact centrality calculation failed"
        )
    print(
        f"[data] retained {int(audit.get('final_retained', 0))} "
        "fundamental periodic graphs"
    )


# apply the fixed graph size and radius-to-length validity criteria
def _passes_quality_filters(graph: Dict[str, object], strut_radius: float) -> Tuple[bool, str]:
    max_joints = _optional_positive_int("DATA_FILTER_MAX_JOINTS") if _cfg_bool("DATA_FILTER_USE_MAX_JOINTS", True) else None
    max_struts = _optional_positive_int("DATA_FILTER_MAX_STRUTS") if _cfg_bool("DATA_FILTER_USE_MAX_STRUTS", True) else None
    max_r_over_l = (
        _optional_positive_float("DATA_FILTER_MAX_RADIUS_LENGTH_RATIO")
        if _cfg_bool("DATA_FILTER_USE_MAX_RADIUS_LENGTH_RATIO", True)
        else None
    )

    n_joints = int(graph["num_joints"])
    n_struts = int(graph["num_struts"])
    if n_joints == 0:
        return False, "no_joints"
    if max_joints is not None and n_joints > max_joints:
        return False, "too_many_joints"
    if max_struts is not None and n_struts > max_struts:
        return False, "too_many_struts"
    if max_r_over_l is not None:
        vectors = np.asarray(graph["strut_vectors"], dtype=np.float64)
        if vectors.size == 0:
            return False, "no_struts"
        lengths = np.linalg.norm(vectors, axis=1)
        min_length = float(np.min(lengths)) if lengths.size else 0.0
        if min_length <= 1e-12:
            return False, "zero_length_strut"
        if float(strut_radius) / min_length >= max_r_over_l:
            return False, "not_slender"

    length_relative_error = float(
        graph.get("catalog_fundamental_length_relative_error", math.inf)
    )
    if (
        not math.isfinite(length_relative_error)
        or length_relative_error >= CATALOG_FPS_LENGTH_RELATIVE_TOLERANCE
    ):
        return False, "catalog_fps_length_mismatch"

    return True, ""

# return paired tensor and metadata paths for one cache key
def _cache_paths(project_root: Path, cache_key: str) -> Tuple[Path, Path]:
    d = _cache_dir(project_root)
    return d / f"{cache_key}.pt", d / f"{cache_key}.meta.json"

# remove superseded cache pairs while retaining the active key
def _remove_old_caches(project_root: Path, keep_key: str) -> None:
    d = _cache_dir(project_root)
    keep_names = {f"{keep_key}.pt", f"{keep_key}.meta.json"}
    for path in d.iterdir():
        if path.name in keep_names:
            continue
        if path.suffix == ".pt" or path.name.endswith(".meta.json"):
            try:
                path.unlink()
                print(f"[data] removed old dataset cache: {path.name}")
            except Exception as e:
                print(f"[data] could not remove old dataset cache {path.name}: {e}")

# public dataset loader
# take one deterministic fraction before grouped dataset splitting
def _sample_dataset_fraction(
    hetero_items,
    homog_items,
    normalizers,
    fraction: float,
):
    if fraction >= 1.0 or len(hetero_items) == 0:
        return hetero_items, homog_items, dict(normalizers or {})
    if len(hetero_items) != len(homog_items):
        raise RuntimeError("heterogeneous and homogeneous dataset sizes differ")

    rng = np.random.default_rng(int(CFG["SPLIT_SEED"]))
    n = max(1, int(round(fraction * len(hetero_items))))
    idxs = rng.choice(len(hetero_items), size=n, replace=False).tolist()
    names = list((normalizers or {}).get("names", []))
    sampled_normalizers = dict(normalizers or {})
    if len(names) == len(hetero_items):
        sampled_normalizers["names"] = [names[i] for i in idxs]

    sampled_hetero = [hetero_items[i] for i in idxs]
    sampled_homog = [homog_items[i] for i in idxs]
    print(
        f"[data] sampled FRACTION={fraction:.3f}: "
        f"{len(hetero_items)} -> {len(sampled_hetero)} structures"
    )
    return sampled_hetero, sampled_homog, sampled_normalizers

# return heterogeneous data, homogeneous data, feature dimensions, and normalization metadata
# normalization metadata includes training, validation, and test indices when normalize is true
# normalize=False returns unnormalized cached features for a caller-defined split
# fraction_override replaces the configured dataset fraction when provided
def load_datasets(
    normalize: bool = True,
    fraction_override: float | None = None,
    rebuild_if_missing: bool = True,
):
    set_global_seed(int(CFG["SPLIT_SEED"]))
    fraction = float(CFG["FRACTION"] if fraction_override is None else fraction_override)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("dataset fraction must be in (0, 1]")

    project_root = Path(CFG.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
    cfg_path = CFG.get("CATALOG_PATH", None)
    catalog_path = Path(cfg_path) if cfg_path else None
    if not catalog_path or not Path(catalog_path).is_file():
        alt = _find_catalog(project_root)
        if alt is not None:
            catalog_path = alt
        else:
            raise FileNotFoundError(
                "no valid Unit_Cell_Catalog.txt found. set CATALOG_PATH in config, "
                f"or place 'Unit_Cell_Catalog.txt' under {project_root} or a 'data/' folder."
            )
    else:
        catalog_path = Path(catalog_path)

    # give viz.py immediate access to the parsed catalog
    _ensure_structures(catalog_path)

    # try cache first
    key = _cache_key(catalog_path)
    pt_path, meta_path = _cache_paths(project_root, key)
    if pt_path.exists() and meta_path.exists():
        try:
            pack = torch.load(pt_path, map_location="cpu", weights_only=False)
            with meta_path.open("r") as f:
                cache_meta = json.load(f)

            hetero_items = pack["hetero"]
            homog_items = pack["homog"]
            dims_meta = pack["dims_meta"]
            normalizers = pack["normalizers"]
            saved_filter_audit = cache_meta.get("filter_audit", None)
            if isinstance(saved_filter_audit, dict):
                _print_filter_audit(saved_filter_audit, cached=True)
            print("[data] reused existing dataset cache")

            # ensure every heterogeneous sample has root level global features
            fixed = 0
            for h in hetero_items:
                if not hasattr(h, "u"):
                    h.u = torch.zeros((1, 0), dtype=torch.float32)
                    fixed += 1
            if fixed:
                print(f"[data] patched missing root 'u' on {fixed} cached samples")

            hetero_items, homog_items, normalizers = _sample_dataset_fraction(
                hetero_items,
                homog_items,
                normalizers,
                fraction,
            )

            if not normalize:
                return hetero_items, homog_items, dims_meta, dict(normalizers or {})

            names_now = list((normalizers or {}).get("names", []))
            if len(names_now) != len(hetero_items):
                names_now = None
            normalizers = _set_splits_and_normalize(hetero_items, homog_items, normalizers, names_now)

            return hetero_items, homog_items, dims_meta, normalizers

        except Exception as e:
            print(f"[data] cache load failed, rebuilding… ({e})")

    if not rebuild_if_missing:
        raise FileNotFoundError(
            "no compatible dataset cache was found. run the combined cache-and-sweep "
            "Slurm script before sweep or final evaluation."
        )

    # rebuild when no compatible cache is available
    print("[data] reading Lumpe and Stankovic (2021) catalog ...")
    raw, parser_counts = _parse_catalog_eth_style(
        catalog_path,
        return_stats=True,
    )
    _print_catalog_preprocessing_counts(parser_counts)

    want = str(CFG["DATASET_FILTER"]).strip()
    if want.lower() != "all":
        pref_map = {
            "cubic": "cub",
            "hexagonal": "hex",
            "orthorhombic": "ort",
            "tetragonal": "tet",
            "trigonal": "trig",
            "monoclinic": "mon",
        }
        pref = pref_map.get(want.lower())
        if pref:
            before = len(raw)
            raw = {k: v for k, v in raw.items() if k.lower().startswith(pref)}
            print(f"[data] dataset filter '{want}' → kept {len(raw)}/{before}")
        else:
            print(f"[data] warning: unknown DATASET_FILTER='{want}', keeping all")

    names = sorted(raw.keys())
    if len(names) == 0:
        try:
            with open(catalog_path, "r", encoding="utf-8", errors="replace") as f:
                lines = [ln.rstrip("\n") for ln in f if ln.strip()]
            for ln in lines[:24]:
                print("    ", ln)
        except Exception:
            pass
        raise RuntimeError(
            "no entries found. this parser expects the ETH/RCSR/EPINET text with blocks split by 89 dashes and "
            "markers 'Name:', 'Normalized unit cell parameters', 'Nodal positions:', 'Bar connectivities:', "
            "and 'Ex = …', 'Gyz = …' lines."
        )

    hetero_items, homog_items, metas, built_names = [], [], [], []
    skipped_periodic_reasons = defaultdict(int)
    skipped_quality = defaultdict(int)
    skipped_centrality = defaultdict(int)
    sanitation_action_counts = defaultdict(int)
    fundamental_method_counts = defaultdict(int)
    fundamental_fallback_counts = defaultdict(int)
    rel_density = REL_DENSITY
    strut_radius_mm = STRUT_RADIUS_MM
    for idx, nm in enumerate(names, 1):
        if idx == 1 or idx % 100 == 0 or idx == len(names):
            print(f"[data] building {idx}/{len(names)} ...")
        entry = raw[nm]

        # positions in fractional cell, then build the canonical periodic graph
        pos_frac = np.vstack(entry["pos"]).astype(np.float32)
        a, b, c, alpha, beta, gamma = entry["cell"]
        struts_raw = np.asarray(entry["struts"], dtype=np.int64)
        if struts_raw.ndim != 2 or struts_raw.shape[1] != 2:
            skipped_periodic_reasons["malformed_strut_array"] += 1
            continue
        pos_cart, cell_matrix = _unscaled_cell_geometry(
            pos_frac, (a, b, c, alpha, beta, gamma)
        )

        try:
            graph = _build_periodic_structure_graph(
                pos_frac, pos_cart, struts_raw, cell_matrix
            )
            graph, L_mm = _scale_periodic_graph_to_density(
                graph, cell_matrix, rel_density, strut_radius_mm
            )
        except (AssertionError, IndexError, ValueError) as e:
            skipped_periodic_reasons[
                _classify_periodic_failure(e)
            ] += 1
            continue
        if int(graph.get("grega_nodes_on_edges_split", 0)) > 0:
            sanitation_action_counts[
                "structures_with_nodes_on_struts_split"
            ] += 1
        if int(graph.get("grega_intersecting_edges_split", 0)) > 0:
            sanitation_action_counts[
                "structures_with_intersections_split"
            ] += 1
        if int(graph.get("grega_window_attempts", 0)) > 1:
            sanitation_action_counts[
                "structures_requiring_window_retry"
            ] += 1
        ok, reason = _passes_quality_filters(graph, strut_radius_mm)
        if not ok:
            skipped_quality[reason] += 1
            continue
        try:
            s_x, j_x, u_x = _build_feature_blocks(
                graph, (a, b, c, alpha, beta, gamma), L_mm
            )
        except CentralityNonConvergenceError:
            skipped_centrality["eigenvector_nonconvergence"] += 1
            continue
        except CentralityComputationError:
            skipped_centrality["other_exact_centrality_failure"] += 1
            continue
        fundamental_method_counts[
            str(graph.get("fundamental_representation_method", "unknown"))
        ] += 1
        fallback_reason = str(
            graph.get("fundamental_representation_fallback_reason", "")
        )
        if fallback_reason.startswith("raw crop has"):
            fundamental_fallback_counts["raw_crop_complexity"] += 1
        elif fallback_reason.startswith(
            "Grega et al. (2025) reconstruction failed"
        ):
            fundamental_fallback_counts["grega_reconstruction_failure"] += 1
        elif fallback_reason.startswith(
            "Grega et al. (2025) sanitation changed"
        ):
            fundamental_fallback_counts["grega_changed_lumpe_geometry"] += 1

        # targets are concatenated (Ex,Ey,Ez,Gyz,Gxz,Gxy)
        targets_raw = np.array(
            entry.get("targets", list(entry["E"]) + list(entry["G"])),
            dtype=np.float32,
        )

        # convert to log10 before training split z-scoring is applied later
        targets = np.log10(np.clip(targets_raw, 1e-9, None)).astype(np.float32)

        h = _pack_hetero(entry, graph, s_x, j_x, u_x, targets)
        g = _pack_homog(entry, graph, s_x, j_x, u_x, targets)

        hetero_items.append(h)
        homog_items.append(g)
        metas.append(_collect_dims_meta(h["strut"].x.numpy(), h["joint"].x.numpy(), h.u.numpy()))
        built_names.append(nm)

    filter_audit = {
        "parser_counts": dict(parser_counts),
        "periodic_attempted": len(names),
        "periodic_rejection_counts": dict(skipped_periodic_reasons),
        "quality_filter_counts": dict(skipped_quality),
        "centrality_rejection_counts": dict(skipped_centrality),
        "sanitation_action_counts": dict(sanitation_action_counts),
        "final_retained": len(hetero_items),
    }
    _print_filter_audit(filter_audit)
    if fundamental_fallback_counts:
        details = ", ".join(
            f"{reason}={count}"
            for reason, count in sorted(fundamental_fallback_counts.items())
        )
        print(f"[data] fundamental periodic structure fallback reasons: {details}")

    # dims are cached with raw features; normalization is split dependent below
    dims_meta = _merge_dims_meta(metas)
    normalizers = {
        "names": built_names,
        "quality_filter_counts": dict(skipped_quality),
        "centrality_rejection_counts": dict(skipped_centrality),
        "filter_audit": filter_audit,
        "fundamental_representation_method_counts": dict(fundamental_method_counts),
        "fundamental_representation_fallback_counts":
            dict(fundamental_fallback_counts),
    }

    # save cache
    pt_path, meta_path = _cache_paths(project_root, key)
    try:
        torch.save(
            {"hetero": hetero_items, "homog": homog_items, "dims_meta": dims_meta, "normalizers": normalizers},
            pt_path
        )
        with meta_path.open("w") as f:
            json.dump(
                {
                    "saved": True,
                    "count": len(hetero_items),
                    "filter_audit": filter_audit,
                    "fundamental_representation_method_counts":
                        dict(fundamental_method_counts),
                    "fundamental_representation_fallback_counts":
                        dict(fundamental_fallback_counts),
                },
                f,
            )
        print(f"[data] saved dataset cache: {pt_path.name}")
        _remove_old_caches(project_root, key)
    except Exception as e:
        print(f"[data] cache save skipped ({e})")

    hetero_items, homog_items, normalizers = _sample_dataset_fraction(
        hetero_items,
        homog_items,
        normalizers,
        fraction,
    )

    if not normalize:
        return hetero_items, homog_items, dims_meta, dict(normalizers or {})

    names_now = list((normalizers or {}).get("names", []))
    if len(names_now) != len(hetero_items):
        names_now = None
    normalizers = _set_splits_and_normalize(hetero_items, homog_items, normalizers, names_now)

    return hetero_items, homog_items, dims_meta, normalizers
