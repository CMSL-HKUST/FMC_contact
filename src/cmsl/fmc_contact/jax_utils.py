"""Ccontact simulation based on Fiber Monte Carlo.

Based on Richardson et al.'s work in "Fiber Monte Carlo":
- https://github.com/PrincetonLIPS/fibermc
- https://openreview.net/forum?id=sP1tCl2QBk

Adapted for JAX-FEM  and contact mechanics simulations.

"""
import numpy as static_np
import jax
import jax.numpy as np
import jax.tree_util as tree_util
from typing import List, TypeVar
pytree: type = TypeVar("Pytree")
FP32: type = np.float32
FP64: type = np.float64


def merge_trees(trees: List[pytree]) -> pytree:
    merged: pytree = trees[0]
    for tree in trees[1:]:
        merged: pytree = tree_util.tree_map(
            lambda t, a: np.vstack((t, a)), merged, tree
        )
    return merged


def index_trees(trees: pytree, index: int) -> pytree:
    return tree_util.tree_map(lambda x: x[index], trees)


def divide_pytree(tree: pytree, divisor: float) -> pytree:
    return tree_util.tree_map(lambda pt: pt / divisor, tree)


def add_pytrees(first_pytree: pytree, second_pytree: pytree) -> pytree:
    return tree_util.tree_map(
        lambda first_tree, second_tree: first_tree + second_tree,
        first_pytree,
        second_pytree,
    )


def tree_stack(trees: List[pytree]):

    leaves_list = []
    treedef_list = []
    for tree in trees:
        leaves, treedef = tree_util.tree_flatten(tree)
        leaves_list.append(leaves)
        treedef_list.append(treedef)

    grouped_leaves = zip(*leaves_list)
    result_leaves = [np.stack(l) for l in grouped_leaves]
    return treedef_list[0].unflatten(result_leaves)


def divides(a: int, b: int) -> bool:
    return a % b == 0


def array_size(arr: np.ndarray) -> int:
    # TODO unnecessary
    return static_np.array(arr).nbytes


def vectorize(signature: str, excluded: tuple = ()) -> callable:
    def decorator(f: callable) -> callable:
        vectorized: callable = np.vectorize(f, excluded=excluded, signature=signature)
        return vectorized

    return decorator


def jit_vectorize(signature: str, excluded: tuple = ()) -> callable:


    def decorator(f: callable) -> callable:
        vectorized: callable = np.vectorize(f, excluded=excluded, signature=signature)
        jitted_and_vectorized: callable = jax.jit(vectorized, static_argnums=excluded)
        return jitted_and_vectorized

    return decorator

def divide00(
    numerator: np.ndarray, denominator: np.ndarray, numeric_type: type = FP32
) -> np.ndarray:

    force_zero: np.ndarray = np.logical_and(numerator == 0, denominator == 0)
    quotient: np.ndarray = np.where(
        force_zero, numeric_type(0.0), numerator
    ) / np.where(force_zero, numeric_type(1.0), denominator)
    return quotient


def zero_one_sign(arr: np.ndarray) -> np.ndarray:

    binary_arr: np.ndarray = 0.5 * (1.0 + np.sign(arr))
    return binary_arr


def vectorized_cond(predicate, true_function, false_function, operand) -> np.ndarray:
    # true_fun and false_fun must act elementwise (i.e. be vectorized)
    true_op = np.where(predicate, operand, 0)
    false_op = np.where(predicate, 0, operand)
    return np.where(predicate, true_function(true_op), false_function(false_op))
