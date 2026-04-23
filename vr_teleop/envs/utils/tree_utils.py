from collections.abc import Sequence
from typing import (
    TypedDict,
    TypeVar,
    Union,  # type: ignore for type alias
)

import numpy as np
import torch

SUPPORTED_TYPES = Union[  # type: ignore
    Sequence, dict, torch.Tensor, np.ndarray, TypedDict, list, tuple, None
]
T = TypeVar("T", bound=SUPPORTED_TYPES)


def stack(trees: Sequence[T], dim: int = 0) -> T:
    tree0 = trees[0]
    if isinstance(tree0, dict):
        return {k: stack([t[k] for t in trees], dim) for k in trees[0]}  # type: ignore
    elif isinstance(tree0, (list, tuple)):
        return [stack([t[i] for t in trees], dim) for i in range(len(trees[0]))]  # type: ignore
    elif isinstance(tree0, np.ndarray):
        return np.stack(trees, axis=dim)  # type: ignore
    elif tree0 is None:
        return None  # type: ignore
    else:
        return torch.stack(trees, dim=dim)  # type: ignore


def index(tree: T, key) -> T:
    match tree:
        case dict():
            return {k: index(v, key) for k, v in tree.items()}  # type: ignore
        case list() | tuple():
            return [index(v, key) for v in tree]  # type: ignore
        case None:
            return None  # type: ignore
        case _:
            return tree[key]  # type: ignore
