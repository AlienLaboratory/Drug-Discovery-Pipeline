"""Parallel processing utilities for batch molecular operations."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, List, TypeVar

from claudedd.core.logging import progress_bar

T = TypeVar("T")
R = TypeVar("R")


def parallel_map(
    func: Callable[[T], R],
    items: List[T],
    n_jobs: int = 1,
    desc: str = "Processing",
    chunk_size: int = 100,
) -> List[R]:
    if n_jobs == 1:
        return [func(item) for item in progress_bar(items, desc=desc)]

    results = [None] * len(items)
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {
            executor.submit(func, item): i for i, item in enumerate(items)
        }
        for future in progress_bar(
            as_completed(futures), total=len(futures), desc=desc
        ):
            idx = futures[future]
            results[idx] = future.result()
    return results
