import time
import os
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List

import torch

logger = logging.getLogger(__name__)


@dataclass
class CycleStats:
    cycle: int
    wall_time_s: float
    peak_gpu_mb: float
    peak_ram_mb: float


@contextmanager
def profile_cycle(cycle: int, stats_list: List[CycleStats]):
    """Profile wall-clock and peak memory for one BO cycle.

    Logs results and appends a CycleStats entry to stats_list.
    A rising GPU curve signals imminent OOM — the plan's key diagnostic.
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        gpu_mb = (
            torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
        )

        try:
            import psutil
            ram_mb = psutil.Process(os.getpid()).memory_info().rss / 1e6
        except ImportError:
            ram_mb = 0.0

        stats_list.append(CycleStats(cycle, elapsed, gpu_mb, ram_mb))

        if gpu_mb > 0:
            logger.info(
                f"Cycle {cycle}: {elapsed:.2f}s  GPU {gpu_mb:.0f}MB  RAM {ram_mb:.0f}MB"
            )
        else:
            logger.info(f"Cycle {cycle}: {elapsed:.2f}s  RAM {ram_mb:.0f}MB")
