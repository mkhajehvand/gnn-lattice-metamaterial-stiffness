# apply the Grega et al. (2025) reconstruction within this project
# geometric cleanup and periodic reconstruction use the adapted implementation in _grega_lattice.py
# this file handles validation, deterministic indexing, strut adjacency multiplicities, and conversion to the graph format used in this project

from __future__ import annotations

from collections import defaultdict
import hashlib
from typing import Dict, List, Tuple

import numpy as np

from ._grega_lattice import Lattice, PeriodicPartnersError, WindowingError


# raised when the required fundamental periodic structure cannot be produced
class FundamentalRepresentationError(ValueError):
    pass


# raised when Grega et al. (2025) geometry checks reject an input
class AuthorGeometryRejectedError(FundamentalRepresentationError):
    pass


AUTHOR_MIN_NODE_DISTANCE = 5e-2
AUTHOR_MIN_EDGE_LENGTH = 5e-3
AUTHOR_SHIFT_FEASIBILITY_ATTEMPTS = 3
AUTHOR_WINDOW_ATTEMPTS = 10


# return a stable seed without coupling preprocessing to training RNG state
def _geometry_seed(
    pos_frac: np.ndarray,
    struts: np.ndarray,
    cell_matrix: np.ndarray,
) -> int:
    digest = hashlib.sha256()
    for array in (
        # the Lumpe and Stankovic (2021) catalog stores fractional coordinates to five decimal places
        # normalize at that precision so float32 conversion does not select a different random window for the same stored geometry
        np.round(np.asarray(pos_frac, dtype=np.float64), 5),
        np.asarray(struts, dtype=np.int64),
        np.round(np.asarray(cell_matrix, dtype=np.float64), 8),
    ):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(contiguous.tobytes())
    return int.from_bytes(digest.digest()[:4], "little", signed=False)


# construct multiplicity aware adjacencies between periodic struts
def _line_graph_adjacency(
    endpoints: np.ndarray,
) -> List[Tuple[int, int, int]]:
    by_joint: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for sid, (u_raw, v_raw) in enumerate(
        np.asarray(endpoints, dtype=np.int64).reshape(-1, 2)
    ):
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
            for sid_b in strut_ids[ai + 1 :]:
                multiplicities[(sid_a, sid_b)] += (
                    count_a * int(counts[sid_b])
                )
    return [
        (int(a), int(b), int(multiplicity))
        for (a, b), multiplicity in sorted(multiplicities.items())
        if int(multiplicity) > 0
    ]


# return the ordinary pair distance used by Grega et al. (2025)
def _minimum_node_distance(positions: np.ndarray) -> float:
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    if len(positions) < 2:
        return float("inf")

    minimum_squared = float("inf")
    for index in range(len(positions) - 1):
        differences = positions[index + 1 :] - positions[index]
        squared = np.einsum(
            "ij,ij->i", differences, differences, optimize=True
        )
        local_minimum = float(np.min(squared))
        minimum_squared = min(minimum_squared, local_minimum)
    return float(np.sqrt(minimum_squared))


# apply the Grega et al. (2025) reduced coordinate 0.05 cutoff
def validate_author_node_distance(
    positions: np.ndarray,
    *,
    stage: str,
) -> float:
    minimum_distance = _minimum_node_distance(positions)
    if minimum_distance < AUTHOR_MIN_NODE_DISTANCE:
        raise AuthorGeometryRejectedError(
            "minimum reduced-coordinate node distance "
            f"{minimum_distance:.8g} is below the Grega et al. (2025) "
            f"{AUTHOR_MIN_NODE_DISTANCE:.8g} cutoff {stage}"
        )
    return float(minimum_distance)


# apply the released one-pass sanitation sequence and project safety checks
def _sanitize_lattice(
    lat: Lattice,
    *,
    enforce_min_node_separation: bool = True,
) -> Tuple[int, int, int, float, float]:
    if enforce_min_node_separation:
        minimum_distance_before = validate_author_node_distance(
            lat.reduced_node_coordinates,
            stage="before sanitation",
        )
    else:
        minimum_distance_before = _minimum_node_distance(
            lat.reduced_node_coordinates
        )
    node_edge_splits = 0
    intersection_splits = 0
    modified = False
    try:
        nodes_on_edges = lat.find_nodes_on_edges()
        if nodes_on_edges:
            node_edge_splits = len(nodes_on_edges)
            lat.split_edges_by_points(nodes_on_edges)
            modified = True

        edge_intersections = lat.find_edge_intersections()
        if edge_intersections:
            intersection_splits = len(edge_intersections)
            lat.split_edges_by_points(edge_intersections)
            modified = True
    except (AssertionError, IndexError, ValueError) as exc:
        raise FundamentalRepresentationError(
            f"Grega et al. (2025) sanitation failed: {exc or type(exc).__name__}"
        ) from exc

    if enforce_min_node_separation:
        minimum_distance_after = validate_author_node_distance(
            lat.reduced_node_coordinates,
            stage="after sanitation",
        )
    else:
        minimum_distance_after = _minimum_node_distance(
            lat.reduced_node_coordinates
        )
    if modified:
        try:
            remaining_nodes_on_edges = lat.find_nodes_on_edges()
            remaining_intersections = (
                []
                if remaining_nodes_on_edges
                else lat.find_edge_intersections()
            )
        except (AssertionError, IndexError, ValueError) as exc:
            raise FundamentalRepresentationError(
                "Grega et al. (2025) sanitation safety check failed: "
                f"{exc or type(exc).__name__}"
            ) from exc
        if remaining_nodes_on_edges or remaining_intersections:
            raise FundamentalRepresentationError(
                "nodes on edges or edge intersections remain after the "
                "released one-pass sanitation"
            )

    return (
        int(node_edge_splits),
        int(intersection_splits),
        0,
        float(minimum_distance_before),
        float(minimum_distance_after),
    )


# run the released feasibility check, windowing, and tracing sequence
def _window_and_trace(
    lat: Lattice,
    seed: int,
    max_window_attempts: int = AUTHOR_WINDOW_ATTEMPTS,
) -> Tuple[Lattice, int, int, np.ndarray]:
    previous_random_state = np.random.get_state()
    try:
        # the released prepare_data.py first performs a three-attempt feasibility check and then independently creates one window with the library default of ten attempts
        # a stable per geometry seed makes that stochastic sequence reproducible without changing its logic
        np.random.seed(int(seed))
        _, feasibility_attempts = lat.obtain_shift_vector(
            max_num_attempts=AUTHOR_SHIFT_FEASIBILITY_ATTEMPTS,
            min_edge_length=AUTHOR_MIN_EDGE_LENGTH,
            return_attempts=True,
        )
        create_window_state = np.random.get_state()
        windowed, window_attempts = lat.create_windowed(
            max_num_attempts=int(max_window_attempts),
            min_edge_length=AUTHOR_MIN_EDGE_LENGTH,
            return_attempts=True,
        )
        if int(window_attempts) == 0:
            window_shift = np.zeros(3, dtype=np.float64)
        else:
            replay_rng = np.random.RandomState()
            replay_rng.set_state(create_window_state)
            window_shift = np.zeros(3, dtype=np.float64)
            for _ in range(int(window_attempts)):
                window_shift = (
                    replay_rng.rand(3)
                    * np.asarray(lat.UC_L, dtype=np.float64)
                )
        windowed.calculate_fundamental_representation()
        return (
            windowed,
            int(feasibility_attempts),
            int(window_attempts),
            window_shift,
        )
    except (
        WindowingError,
        PeriodicPartnersError,
        AssertionError,
        IndexError,
        ValueError,
    ) as exc:
        raise FundamentalRepresentationError(
            "Grega et al. (2025) periodic windowing/fundamental tracing "
            f"failed: {exc or type(exc).__name__}"
        ) from exc
    finally:
        np.random.set_state(previous_random_state)


# canonicalize upstream fundamental nodes and labeled periodic edges
def _canonical_fundamental_edges(
    positions: np.ndarray,
    fundamental_edges: np.ndarray,
    tessellation_vectors: np.ndarray,
    cell_matrix: np.ndarray,
) -> Tuple[
    List[tuple],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[Tuple[int, ...]],
    Dict[int, int],
]:
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    upstream_edges = np.asarray(
        fundamental_edges, dtype=np.int64
    ).reshape(-1, 2)
    tessellation_vectors = np.asarray(
        tessellation_vectors, dtype=np.float64
    )
    if tessellation_vectors.ndim != 2 or tessellation_vectors.shape[0] != len(
        upstream_edges
    ):
        raise FundamentalRepresentationError(
            "fundamental tessellation vectors do not match fundamental edges"
        )
    if tessellation_vectors.shape[1] == 6:
        upstream_shifts = (
            tessellation_vectors[:, 3:] - tessellation_vectors[:, :3]
        )
    elif tessellation_vectors.shape[1] == 3:
        upstream_shifts = tessellation_vectors
    else:
        raise FundamentalRepresentationError(
            "fundamental tessellation vectors must have width 3 or 6"
        )

    upstream_joint_ids = sorted(
        int(x) for x in np.unique(upstream_edges).tolist()
    )
    upstream_to_local = {
        upstream_id: local_id
        for local_id, upstream_id in enumerate(upstream_joint_ids)
    }
    joint_positions = positions[upstream_joint_ids]
    key_order = [
        tuple(np.round(position, 12).tolist()) for position in joint_positions
    ]

    records = []
    for sid, ((upstream_u, upstream_v), upstream_shift) in enumerate(
        zip(upstream_edges, upstream_shifts, strict=True)
    ):
        u = upstream_to_local[int(upstream_u)]
        v = upstream_to_local[int(upstream_v)]
        frac_vector = (
            positions[int(upstream_v)]
            - positions[int(upstream_u)]
            + np.asarray(upstream_shift, dtype=np.float64)
        )
        if u > v:
            u, v = v, u
            frac_vector = -frac_vector
        elif u == v:
            for component in frac_vector:
                if abs(float(component)) > 1e-12:
                    if float(component) < 0.0:
                        frac_vector = -frac_vector
                    break

        base_delta = joint_positions[v] - joint_positions[u]
        shift_float = frac_vector - base_delta
        shift = np.rint(shift_float).astype(np.int64)
        if not np.allclose(shift_float, shift, atol=5e-6, rtol=0.0):
            raise FundamentalRepresentationError(
                "upstream fundamental edge has a non-integer cell shift"
            )
        exact_frac_vector = base_delta + shift
        vector = (
            np.asarray(cell_matrix, dtype=np.float64).reshape(3, 3)
            @ exact_frac_vector
        )
        records.append(
            (
                int(u),
                int(v),
                *shift.tolist(),
                int(sid),
                exact_frac_vector,
                vector,
            )
        )

    records.sort(key=lambda record: record[:6])
    unique_records = []
    upstream_strut_groups: List[List[int]] = []
    for record in records:
        label = record[:5]
        if unique_records and unique_records[-1][:5] == label:
            upstream_strut_groups[-1].append(int(record[5]))
            continue
        unique_records.append(record)
        upstream_strut_groups.append([int(record[5])])
    records = unique_records
    endpoints = np.asarray(
        [(record[0], record[1]) for record in records], dtype=np.int64
    ).reshape(-1, 2)
    shifts = np.asarray(
        [record[2:5] for record in records], dtype=np.int64
    ).reshape(-1, 3)
    fractional_vectors = np.asarray(
        [record[6] for record in records], dtype=np.float32
    ).reshape(-1, 3)
    vectors = np.asarray(
        [record[7] for record in records], dtype=np.float32
    ).reshape(-1, 3)
    upstream_strut_ids = [
        tuple(sorted(group)) for group in upstream_strut_groups
    ]

    return (
        key_order,
        joint_positions,
        endpoints,
        vectors,
        fractional_vectors,
        shifts,
        upstream_strut_ids,
        upstream_to_local,
    )


# build the fundamental periodic graph using Grega et al. (2025)
def build_grega_fundamental_representation(
    pos_frac: np.ndarray,
    struts_raw: np.ndarray,
    cell_matrix: np.ndarray,
    *,
    max_window_attempts: int = AUTHOR_WINDOW_ATTEMPTS,
    enforce_min_node_separation: bool = True,
) -> Dict[str, object]:
    pos_frac = np.asarray(pos_frac, dtype=np.float64).reshape(-1, 3)
    struts_raw = np.asarray(struts_raw, dtype=np.int64).reshape(-1, 2)
    cell_matrix = np.asarray(cell_matrix, dtype=np.float64).reshape(3, 3)
    if len(pos_frac) == 0 or len(struts_raw) == 0:
        raise FundamentalRepresentationError(
            "fundamental periodic structure requires at least one joint and one strut"
        )

    seed = _geometry_seed(pos_frac, struts_raw, cell_matrix)
    lat = Lattice(
        name=f"geometry_{seed:08x}",
        nodal_positions=pos_frac.copy(),
        edge_adjacency=struts_raw.copy(),
    )
    (
        split_node_edges,
        split_intersection_edges,
        unresolved_node_edges,
        minimum_node_distance_before,
        minimum_node_distance_after,
    ) = _sanitize_lattice(
        lat,
        enforce_min_node_separation=enforce_min_node_separation,
    )
    sanitized_num_nodes = int(lat.num_nodes)
    sanitized_num_edges = int(lat.num_edges)
    (
        windowed,
        feasibility_attempts,
        window_attempts,
        window_shift,
    ) = _window_and_trace(
        lat,
        seed=seed,
        max_window_attempts=max_window_attempts,
    )
    minimum_windowed_edge_length = float(
        np.min(windowed.calculate_edge_lengths())
    )

    (
        key_order,
        joint_positions,
        endpoints,
        vectors,
        fractional_vectors,
        shifts,
        upstream_strut_ids,
        upstream_to_local,
    ) = _canonical_fundamental_edges(
        windowed.reduced_node_coordinates,
        windowed.fundamental_edge_adjacency,
        windowed.fundamental_tesselation_vecs,
        cell_matrix,
    )

    fundamental_edge_map = np.asarray(
        windowed.fundamental_edge_map, dtype=np.int64
    )
    source_ids = []
    for upstream_group in upstream_strut_ids:
        windowed_sources = {
            int(x)
            for upstream_sid in upstream_group
            for x in np.flatnonzero(
                fundamental_edge_map == int(upstream_sid)
            )
        }
        source_ids.append(tuple(sorted(windowed_sources)))
    reconstructed = [len(sources) > 1 for sources in source_ids]
    strut_line_adjacency = _line_graph_adjacency(endpoints)
    joint_edges = sorted(
        {
            tuple(sorted((int(u), int(v))))
            for u, v in endpoints
            if int(u) != int(v)
        }
    )

    window_node_types = windowed.calculate_node_types()
    face_nodes = set(int(x) for x in window_node_types["face_nodes"])
    pos_cart = joint_positions @ cell_matrix.T
    graph = {
        "pos_frac": joint_positions.astype(np.float32),
        "pos_cart": pos_cart.astype(np.float32),
        "cell_matrix_unscaled": cell_matrix.astype(np.float32),
        "raw_struts": struts_raw.copy(),
        "virtual_joints": face_nodes,
        "boundary_degree1_joints": face_nodes,
        "boundary_cut_joint_ids": set(),
        "retained_boundary_joint_ids": set(),
        "struts": endpoints,
        "strut_vectors": vectors,
        "strut_fractional_vectors": fractional_vectors,
        "strut_shifts": shifts,
        "strut_sources": source_ids,
        "strut_is_reconstructed": reconstructed,
        "retained_boundary_cut_joints": set(),
        "key_order": key_order,
        "groups": [
            [int(upstream_id)]
            for upstream_id in sorted(upstream_to_local)
        ],
        "joint_map": {
            int(upstream_id): int(local_id)
            for upstream_id, local_id in upstream_to_local.items()
        },
        "strut_endpoints": endpoints,
        "joint_edges": joint_edges,
        "strut_line_edges": [
            (int(a), int(b)) for a, b, _ in strut_line_adjacency
        ],
        "strut_line_adjacency": strut_line_adjacency,
        "num_joints": int(len(key_order)),
        "num_struts": int(len(endpoints)),
        "fundamental_representation_method":
            "grega_prepare_data_fundamental_v4",
        "grega_geometry_seed": int(seed),
        "grega_min_node_distance": float(AUTHOR_MIN_NODE_DISTANCE),
        "grega_min_edge_length": float(AUTHOR_MIN_EDGE_LENGTH),
        "grega_minimum_node_distance_before":
            minimum_node_distance_before,
        "grega_minimum_node_distance_after":
            minimum_node_distance_after,
        "grega_minimum_windowed_edge_length":
            minimum_windowed_edge_length,
        "grega_sanitized_num_nodes": sanitized_num_nodes,
        "grega_sanitized_num_edges": sanitized_num_edges,
        "grega_windowed_num_nodes": int(windowed.num_nodes),
        "grega_windowed_num_edges": int(windowed.num_edges),
        "grega_shift_feasibility_attempts": int(feasibility_attempts),
        "grega_window_attempts": int(window_attempts),
        "grega_window_retry_index": 0,
        "grega_window_shift_vector": window_shift.astype(np.float64),
        "grega_window_split_fallback": False,
        "grega_nodes_on_edges_split": int(split_node_edges),
        "grega_intersecting_edges_split": int(split_intersection_edges),
        "grega_nodes_on_edges_after_sanitation": int(
            unresolved_node_edges
        ),
    }
    return graph
