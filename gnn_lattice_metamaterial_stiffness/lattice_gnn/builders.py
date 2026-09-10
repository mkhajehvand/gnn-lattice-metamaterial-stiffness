# create normalized loaders with aligned splits and optional augmentation

from __future__ import annotations
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torch_geometric.loader import DataLoader

from .data import (
    CYCLIC_TARGET_PERMUTATIONS,
    _real_sh_dir,
    _set_splits_and_normalize,
    load_datasets,
    set_global_seed,
)
from .config import CFG, DATALOADER_NUM_WORKERS

# simple in-process memo so we read the cache only once per python run
_DATASETS_MEMO = None

# seed Python and NumPy from PyTorch's deterministic worker seed.
def _seed_worker(worker_id: int) -> None:
    del worker_id
    import random
    import numpy as np

    worker_seed = int(torch.initial_seed()) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)

# load one full-feature cache and reuse it within the current process
def _get_datasets_once():
    global _DATASETS_MEMO
    if _DATASETS_MEMO is None:
        # cache raw full-feature graphs once; each ablation is cloned and normalized separately
        previous_strut = bool(CFG.get("STRUT_USE_CENTRALITY", True))
        previous_joint = bool(CFG.get("JOINT_USE_CENTRALITY", True))
        centrality_keys = (
            "INCLUDE_DEGREE_CENTRALITY",
            "INCLUDE_CLOSENESS_CENTRALITY",
            "INCLUDE_BETWEENNESS_CENTRALITY",
            "INCLUDE_EIGENVECTOR_CENTRALITY",
        )
        previous_individual = {key: bool(CFG.get(key, True)) for key in centrality_keys}
        CFG["STRUT_USE_CENTRALITY"] = True
        CFG["JOINT_USE_CENTRALITY"] = True
        for key in centrality_keys:
            CFG[key] = True
        try:
            _DATASETS_MEMO = load_datasets(
                normalize=False,
                rebuild_if_missing=False,
            )
        finally:
            CFG["STRUT_USE_CENTRALITY"] = previous_strut
            CFG["JOINT_USE_CENTRALITY"] = previous_joint
            CFG.update(previous_individual)
    return _DATASETS_MEMO

# move selected centralities from encoder inputs to pooling weights
def _select_centrality_features(
    hetero_items: List,
    homog_items: List,
    dims_meta: dict,
) -> dict:
    switches = (
        "INCLUDE_DEGREE_CENTRALITY",
        "INCLUDE_CLOSENESS_CENTRALITY",
        "INCLUDE_BETWEENNESS_CENTRALITY",
        "INCLUDE_EIGENVECTOR_CENTRALITY",
    )
    selected = [index for index, key in enumerate(switches) if bool(CFG.get(key, True))]
    strut_width = int(dims_meta.get("s_centrality_dim", 0))
    joint_width = int(dims_meta.get("j_centrality_dim", 0))
    if strut_width != 4 or joint_width != 4:
        raise RuntimeError(
            "full-feature dataset cache must contain four joint and strut "
            "centrality features"
        )
    for hetero, homogeneous in zip(hetero_items, homog_items):
        strut_prefix = hetero["strut"].x[:, :-strut_width]
        joint_prefix = hetero["joint"].x[:, :-joint_width]
        strut_centrality = hetero["strut"].x[:, -strut_width:].clone()
        joint_centrality = hetero["joint"].x[:, -joint_width:].clone()
        edge_prefix = homogeneous.edge_attr[:, :-strut_width]
        homogeneous_joint_prefix = homogeneous.x[:, :-joint_width]
        homogeneous_strut_centrality = homogeneous.edge_attr[:, -strut_width:].clone()
        homogeneous_joint_centrality = homogeneous.x[:, -joint_width:].clone()

        if homogeneous_strut_centrality.size(0) % 2 != 0:
            raise RuntimeError(
                "homogeneous edge features must contain two entries per strut"
            )
        if not torch.allclose(
            homogeneous_strut_centrality[::2],
            homogeneous_strut_centrality[1::2],
        ):
            raise RuntimeError(
                "oppositely directed copies of one strut have different centralities"
            )

        hetero["strut"].x = strut_prefix
        hetero["joint"].x = joint_prefix
        homogeneous.edge_attr = edge_prefix
        homogeneous.x = homogeneous_joint_prefix

        hetero["strut"].pooling_centrality = (
            strut_centrality[:, selected].clone()
        )
        hetero["joint"].pooling_centrality = (
            joint_centrality[:, selected].clone()
        )
        homogeneous.strut_pooling_centrality = (
            homogeneous_strut_centrality[::2, selected].clone()
        )
        homogeneous.joint_pooling_centrality = (
            homogeneous_joint_centrality[:, selected].clone()
        )

    out = dict(dims_meta)
    out["s_total_dim"] = int(out["s_total_dim"]) - strut_width
    out["j_total_dim"] = int(out["j_total_dim"]) - joint_width
    out["s_centrality_dim"] = len(selected)
    out["j_centrality_dim"] = len(selected)
    out["centrality_pooling_dim"] = len(selected)
    return out

# replace centrality contrasts by an equal-width zero control
def _remove_centrality_features(
    hetero_items: List,
    homog_items: List,
    dims_meta: dict,
) -> dict:
    strut_width = int(dims_meta.get("s_centrality_dim", 0))
    joint_width = int(dims_meta.get("j_centrality_dim", 0))
    if strut_width != joint_width:
        raise RuntimeError("joint and strut pooling centrality widths differ")
    if len(hetero_items) != len(homog_items):
        raise RuntimeError("heterogeneous and homogeneous dataset sizes differ")

    for hetero, homogeneous in zip(hetero_items, homog_items):
        hetero["strut"].pooling_centrality = torch.zeros_like(
            hetero["strut"].pooling_centrality
        )
        hetero["joint"].pooling_centrality = torch.zeros_like(
            hetero["joint"].pooling_centrality
        )
        homogeneous.strut_pooling_centrality = torch.zeros_like(
            homogeneous.strut_pooling_centrality
        )
        homogeneous.joint_pooling_centrality = torch.zeros_like(
            homogeneous.joint_pooling_centrality
        )

    out = dict(dims_meta)
    out["centrality_pooling_mode"] = "ordinary_pooling_control"
    return out

# clone cached graphs, configure pooling, split, and standardize features
def _prepare_datasets(include_centrality: bool):
    raw_hetero, raw_homog, raw_dims, raw_normalizers = _get_datasets_once()
    hetero_items = [item.clone() for item in raw_hetero]
    homog_items = [item.clone() for item in raw_homog]
    dims_meta = _select_centrality_features(
        hetero_items,
        homog_items,
        dict(raw_dims),
    )
    if not include_centrality and int(dims_meta.get("s_centrality_dim", 0)) > 0:
        dims_meta = _remove_centrality_features(
            hetero_items,
            homog_items,
            dims_meta,
        )

    names = list((raw_normalizers or {}).get("names", []))
    names_for_split = names if len(names) == len(hetero_items) else None
    normalizers = _set_splits_and_normalize(
        hetero_items,
        homog_items,
        dict(raw_normalizers or {}),
        names_for_split,
    )
    normalizers["centrality_features_enabled"] = bool(include_centrality)
    normalizers["centrality_usage"] = "centrality_augmented_pooling"
    normalizers["centrality_pooling_mode"] = (
        "centrality_augmented"
        if include_centrality
        else "ordinary_pooling_control"
    )
    return hetero_items, homog_items, dims_meta, normalizers

# select valid dataset indices without changing their order
def _make_subset(items: List, idxs: List[int]) -> List:
    if not idxs:
        return []
    return [items[i] for i in idxs if 0 <= i < len(items)]

# define the two cyclic axis rotations and corresponding stiffness permutations used to augment non-cubic structures
_CYCLIC_AXIS_PERMUTATIONS = (
    (
        np.asarray(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        CYCLIC_TARGET_PERMUTATIONS[0],
    ),
    (
        np.asarray(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        ),
        CYCLIC_TARGET_PERMUTATIONS[1],
    ),
)

# identify cubic structures, whose cyclic axis copies are equivalent
def _is_cubic_structure_name(name: str) -> bool:
    return str(name).split("_", 1)[0].strip().lower() == "cub"

# permute physical stiffness components before re-standardization
def _permuted_standardized_target(
    target: torch.Tensor,
    permutation: tuple[int, ...],
    normalizers: dict,
) -> torch.Tensor:
    y_stats = normalizers.get("y")
    if not isinstance(y_stats, dict):
        raise ValueError("target statistics are required for training augmentation")
    mean = torch.as_tensor(y_stats["mean"], dtype=target.dtype, device=target.device)
    std = torch.as_tensor(y_stats["std"], dtype=target.dtype, device=target.device)
    order = torch.as_tensor(permutation, dtype=torch.long, device=target.device)
    raw_log_target = target * std + mean
    permuted_raw_log_target = raw_log_target.index_select(-1, order)
    return (permuted_raw_log_target - mean) / std

# rotate one graph pair and its targets under a cyclic axis permutation
def _axis_permuted_training_pair(
    hetero_item,
    homog_item,
    rotation: np.ndarray,
    target_permutation: tuple[int, ...],
    dims_meta: dict,
    normalizers: dict,
):
    hetero_aug = hetero_item.clone()
    homog_aug = homog_item.clone()

    unit_dirs = getattr(hetero_item["strut"], "augmentation_unit_dir", None)
    if unit_dirs is None:
        raise RuntimeError(
            "dataset cache lacks strut directions required for augmentation; "
            "rebuild the dataset cache"
        )
    rotated_dirs = unit_dirs.detach().cpu().numpy() @ rotation.T
    rotated_sh = torch.from_numpy(
        _real_sh_dir(rotated_dirs, lmax=int(CFG["SH_LMAX"]))
    )

    sh_start = int(dims_meta.get("s_length_dim", 0))
    sh_width = int(dims_meta.get("s_sh_dim", 0))
    sh_stop = sh_start + sh_width
    if rotated_sh.shape[1] != sh_width:
        raise ValueError("rotated spherical-harmonic width is inconsistent")
    hetero_aug["strut"].x[:, sh_start:sh_stop] = rotated_sh.to(
        dtype=hetero_aug["strut"].x.dtype
    )

    repeated_sh = rotated_sh.repeat_interleave(2, dim=0)
    if repeated_sh.size(0) != homog_aug.edge_attr.size(0):
        raise ValueError("homogeneous edge order is inconsistent with strut order")
    homog_aug.edge_attr[:, sh_start:sh_stop] = repeated_sh.to(
        dtype=homog_aug.edge_attr.dtype
    )

    target_aug = _permuted_standardized_target(
        hetero_item.y,
        target_permutation,
        normalizers,
    )
    hetero_aug["graph_y"] = target_aug.clone()
    hetero_aug.y = target_aug.clone()
    homog_aug.y = target_aug.clone()
    hetero_aug["strut"].augmentation_unit_dir = torch.from_numpy(
        rotated_dirs.astype(np.float32, copy=False)
    )
    return hetero_aug, homog_aug

# add two cyclic copies of every non-cubic training structure
def _augment_training_items(
    hetero_items: List,
    homog_items: List,
    idx_train: List[int],
    names: List[str],
    dims_meta: dict,
    normalizers: dict,
):
    hetero_train = _make_subset(hetero_items, idx_train)
    homog_train = _make_subset(homog_items, idx_train)
    audit = {
        "enabled": bool(CFG["TRAIN_AXIS_PERMUTATION_AUGMENTATION"]),
        "original_training_samples": len(hetero_train),
        "non_cubic_training_samples": 0,
        "cubic_training_samples_skipped": 0,
        "added_training_samples": 0,
        "final_training_samples": len(hetero_train),
    }
    if not audit["enabled"]:
        print("[data] training axis-permutation augmentation disabled: added 0 training samples")
        return hetero_train, homog_train, audit
    if len(names) != len(hetero_items):
        raise RuntimeError(
            "dataset cache lacks structure names required for non-cubic augmentation; "
            "rebuild the dataset cache"
        )

    augmented_hetero = list(hetero_train)
    augmented_homog = list(homog_train)
    for idx in idx_train:
        if idx < 0 or idx >= len(hetero_items):
            continue
        if _is_cubic_structure_name(names[idx]):
            audit["cubic_training_samples_skipped"] += 1
            continue
        audit["non_cubic_training_samples"] += 1
        for rotation, target_permutation in _CYCLIC_AXIS_PERMUTATIONS:
            hetero_aug, homog_aug = _axis_permuted_training_pair(
                hetero_items[idx],
                homog_items[idx],
                rotation,
                target_permutation,
                dims_meta,
                normalizers,
            )
            augmented_hetero.append(hetero_aug)
            augmented_homog.append(homog_aug)

    audit["added_training_samples"] = len(augmented_hetero) - len(hetero_train)
    audit["final_training_samples"] = len(augmented_hetero)
    print(
        "[data] training axis-permutation augmentation enabled: "
        f"added {audit['added_training_samples']} training samples from "
        f"{audit['non_cubic_training_samples']} non-cubic originals; "
        f"skipped {audit['cubic_training_samples_skipped']} cubic originals; "
        f"training set {audit['original_training_samples']} -> "
        f"{audit['final_training_samples']}; validation/test unchanged; "
        "normalization fitted on original training samples"
    )
    return augmented_hetero, augmented_homog, audit

# create a reproducibly seeded PyG loader with persistent workers.
def _make_loader(items: List, batch_size: int, shuffle: bool) -> DataLoader:
    # keep sampling and worker seeding on separate generators
    # persistent workers consume their generator only on the first iterator, while the sampler consumes its generator every epoch
    seed = int(CFG["TRAIN_SEED"]) & 0xFFFFFFFF
    sampler_gen = torch.Generator()
    sampler_gen.manual_seed(seed)
    worker_gen = torch.Generator()
    worker_gen.manual_seed((seed + 1) & 0xFFFFFFFF)
    sampler = (
        RandomSampler(items, generator=sampler_gen)
        if shuffle
        else SequentialSampler(items)
    )

    num_workers = DATALOADER_NUM_WORKERS
    kwargs = {
        "dataset": items,
        "batch_size": batch_size,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": bool(torch.cuda.is_available()),
        "persistent_workers": num_workers > 0,
        "drop_last": False,
        "generator": worker_gen,
        "worker_init_fn": _seed_worker,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(
        **kwargs,
    )

# reset a reused loader so every trial receives the same sample order
def reset_loader_generator(loader: DataLoader, seed: int) -> None:
    sampler_generator = getattr(getattr(loader, "sampler", None), "generator", None)
    if sampler_generator is not None:
        sampler_generator.manual_seed(int(seed) & 0xFFFFFFFF)
    generator = getattr(loader, "generator", None)
    if generator is not None:
        generator.manual_seed((int(seed) + 1) & 0xFFFFFFFF)

# returns a dict with both hetero and homog families and aliases:
# {
#   'hetero': {'train': dl, 'val': dl, 'test': dl},
#   'homog':  {'train': dl, 'val': dl, 'test': dl},
#   'het':    {'train': dl, 'val': dl, 'test': dl},
#   'hom':    {'train': dl, 'val': dl, 'test': dl},
#   'dims_meta': {...},
#   'normalizers': {...},
# }
def make_loaders(
    include_centrality: bool = True,
) -> Dict[str, Dict[str, DataLoader]]:
    set_global_seed(int(CFG["TRAIN_SEED"]))

    hetero_items, homog_items, dims_meta, normalizers = _prepare_datasets(
        include_centrality=bool(include_centrality)
    )

    idx_train: List[int] = list(normalizers.get("idx_train", []))
    idx_val:   List[int] = list(normalizers.get("idx_val", []))
    idx_test:  List[int] = list(normalizers.get("idx_test", []))

    names = list((normalizers or {}).get("names", []))
    hetero_train, homog_train, augmentation_audit = _augment_training_items(
        hetero_items,
        homog_items,
        idx_train,
        names,
        dims_meta,
        normalizers,
    )
    hetero_val   = _make_subset(hetero_items, idx_val)
    hetero_test  = _make_subset(hetero_items, idx_test)

    homog_val   = _make_subset(homog_items, idx_val)
    homog_test  = _make_subset(homog_items, idx_test)
    normalizers["training_axis_permutation_augmentation"] = augmentation_audit

    bs_train = int(CFG["BATCH_SIZE"])
    bs_eval = bs_train

    hetero = {
        "train": _make_loader(hetero_train, bs_train, shuffle=True),
        "val":   _make_loader(hetero_val,   bs_eval,  shuffle=False),
        "test":  _make_loader(hetero_test,  bs_eval,  shuffle=False),
    }
    homog = {
        "train": _make_loader(homog_train, bs_train, shuffle=True),
        "val":   _make_loader(homog_val,   bs_eval,  shuffle=False),
        "test":  _make_loader(homog_test,  bs_eval,  shuffle=False),
    }

    return {
        "hetero": hetero,
        "homog": homog,
        "het": hetero,
        "hom": homog,
        "dims_meta": dims_meta,
        "normalizers": normalizers,
    }
