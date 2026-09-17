# Copyright (C) 2026
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""
CUDA Graph management for the persistent PyTorch GPU search engine.
"""

import logging
import os
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("pycbc.filter.gpu_search.graphs")

try:
    import torch
except ImportError:
    torch = None

from .core import correlate_and_ifft


class CUDAGraphEntry:
    """A captured CUDA graph with bound input/output device memory references."""

    def __init__(
        self,
        graph: Any,
        stilde_dev: Any,
        cout_workspace: Any,
        out_workspace: Any,
        stream: Any,
        pid: int,
    ):
        self.graph = graph
        self.stilde_dev = stilde_dev
        self.cout_workspace = cout_workspace
        self.out_workspace = out_workspace
        self.stream = stream
        self.pid = pid
        self.replay_count = 0


class CUDAGraphManager:
    """
    Manages creation, caching, replay, and invalidation of CUDA graphs
    for batched template correlation and inverse FFTs.
    """

    def __init__(self, enabled: bool = True, device: str = "cuda"):
        self.enabled = bool(enabled)
        self.device = device
        self._cache: Dict[Tuple, CUDAGraphEntry] = {}
        self._captured_pid = os.getpid()

        self.capture_count = 0
        self.replay_count = 0
        self.invalidation_count = 0

        # Register fork handler if available to prevent child processes from reusing graphs
        if hasattr(os, "register_at_fork"):
            try:
                os.register_at_fork(after_in_child=self._on_fork_child)
            except (RuntimeError, AttributeError):
                pass

    def _on_fork_child(self):
        """Invalidate all cached graphs in forked child process."""
        self.invalidate_all()
        self._captured_pid = os.getpid()

    def is_available(self) -> bool:
        """Check if CUDA graphs are available on current device and PyTorch installation."""
        if not self.enabled or torch is None:
            return False
        if not torch.cuda.is_available():
            return False
        if not hasattr(torch.cuda, "CUDAGraph"):
            return False
        if not str(self.device).startswith("cuda"):
            return False
        return True

    def check_fork_safety(self):
        """Ensure current process ID matches captured process ID."""
        current_pid = os.getpid()
        if current_pid != self._captured_pid:
            logger.warning(
                "Process fork detected (captured PID %d, current PID %d); "
                "invalidating all CUDA graphs.",
                self._captured_pid,
                current_pid,
            )
            self.invalidate_all()
            self._captured_pid = current_pid

    def invalidate_all(self):
        """Invalidate and clear all cached CUDA graphs."""
        count = len(self._cache)
        self._cache.clear()
        self.invalidation_count += 1
        if count > 0:
            logger.info("Invalidated %d cached CUDA graphs.", count)

    def invalidate_tile(self, tile_id: int):
        """Invalidate cached graphs associated with a specific tile ID."""
        keys_to_del = [k for k in self._cache if k[0] == tile_id]
        for k in keys_to_del:
            del self._cache[k]
        if keys_to_del:
            self.invalidation_count += 1
            logger.info(
                "Invalidated %d CUDA graphs for tile %d.",
                len(keys_to_del),
                tile_id,
            )

    def _make_key(
        self,
        tile: Any,
        cout_workspace: Any,
        out_workspace: Any,
        stilde_dev: Any,
        stream: Any,
    ) -> Tuple:
        stream_id = getattr(stream, "cuda_stream", 0) if stream is not None else 0
        dev_idx = (
            torch.cuda.current_device()
            if torch is not None and torch.cuda.is_available()
            else 0
        )
        return (
            tile.tile_id,
            tile.batch_size,
            tile.template_data.data_ptr(),
            cout_workspace.data_ptr(),
            out_workspace.data_ptr(),
            stilde_dev.data_ptr(),
            stream_id,
            dev_idx,
        )

    def get_or_capture(
        self,
        tile: Any,
        cout_workspace: Any,
        out_workspace: Any,
        stilde_dev: Any,
        flen: int,
        tlen: int,
        stream: Optional[Any] = None,
    ) -> Optional[CUDAGraphEntry]:
        """
        Retrieve a cached CUDA graph or capture a new one.
        """
        if not self.is_available():
            return None

        self.check_fork_safety()

        key = self._make_key(
            tile, cout_workspace, out_workspace, stilde_dev, stream
        )
        if key in self._cache:
            return self._cache[key]

        b = tile.batch_size
        use_stream = (
            stream
            if stream is not None
            else torch.cuda.current_stream()
        )

        try:
            # 1. Warmup iteration on target stream
            with torch.cuda.stream(use_stream):
                correlate_and_ifft(
                    templates=tile.template_data,
                    data=stilde_dev,
                    cout_workspace=cout_workspace,
                    out_workspace=out_workspace,
                    tlen=tlen,
                    flen=flen,
                    batch_size=b,
                    stream=use_stream,
                )
            use_stream.synchronize()

            # 2. Capture CUDA graph
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=use_stream):
                correlate_and_ifft(
                    templates=tile.template_data,
                    data=stilde_dev,
                    cout_workspace=cout_workspace,
                    out_workspace=out_workspace,
                    tlen=tlen,
                    flen=flen,
                    batch_size=b,
                    stream=use_stream,
                )

            entry = CUDAGraphEntry(
                graph=g,
                stilde_dev=stilde_dev,
                cout_workspace=cout_workspace,
                out_workspace=out_workspace,
                stream=use_stream,
                pid=os.getpid(),
            )
            self._cache[key] = entry
            self.capture_count += 1
            logger.info(
                "Captured CUDA graph for tile %d (batch=%d, key=%s). Total captured: %d",
                tile.tile_id,
                b,
                key,
                self.capture_count,
            )
            return entry

        except Exception as e:
            logger.warning(
                "CUDA graph capture failed for tile %d: %s. Falling back to eager execution.",
                tile.tile_id,
                e,
            )
            return None

    def replay(
        self,
        entry: CUDAGraphEntry,
        stilde_input: Any,
        stream: Optional[Any] = None,
    ) -> bool:
        """
        Copy new stilde data into static stilde_dev and replay the captured graph.
        """
        self.check_fork_safety()
        if entry.pid != os.getpid():
            self.invalidate_all()
            return False

        use_stream = stream if stream is not None else entry.stream
        with torch.cuda.stream(use_stream):
            if entry.stilde_dev.data_ptr() != stilde_input.data_ptr():
                entry.stilde_dev.copy_(stilde_input)
            entry.graph.replay()

        entry.replay_count += 1
        self.replay_count += 1
        return True
