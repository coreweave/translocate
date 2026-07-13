# SPDX-FileCopyrightText: 2026 CoreWeave
#
# SPDX-License-Identifier: MIT

"""Shared thread pools for the per-file fan-out paths.

Reconciliation's stat-and-hash and materialization's link-or-copy-
and-rehash run per-file over distributions that often hold tens of
thousands of files. They interleave blocking filesystem syscalls with
hashing; both release the GIL, so ``io_pool`` oversubscribes cores to
keep the filesystem busy while a worker waits on a syscall.

The wheel emitter's RECORD pass hashes bytes already held in memory,
with no filesystem wait to overlap. That is CPU work, so more workers
than cores only queue behind busy ones; it runs on ``cpu_pool``,
sized to the core count.

The CLI owns one pool per operation. Tests inject a single-worker
pool when they need deterministic ordering.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

DEFAULT_IO_WORKERS = min(64, (os.cpu_count() or 1) * 4)
DEFAULT_CPU_WORKERS = os.cpu_count() or 1


def io_pool(max_workers: Optional[int] = None) -> ThreadPoolExecutor:
    workers = DEFAULT_IO_WORKERS if max_workers is None else max_workers
    return ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="translocate-io"
    )


def cpu_pool(max_workers: Optional[int] = None) -> ThreadPoolExecutor:
    workers = DEFAULT_CPU_WORKERS if max_workers is None else max_workers
    return ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="translocate-cpu"
    )
