# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ROS 2 executor utilities for asynchronous background spinning."""

from __future__ import annotations

import logging
import threading
from typing import Sequence

import rclpy
from rclpy.executors import (
    Executor,
    ExternalShutdownException,
    ShutdownException,
    SingleThreadedExecutor,
)
from rclpy.node import Node

_logger = logging.getLogger(__name__)


class BackgroundRos2Executor:
    """Manages spinning multiple ROS 2 nodes on a dedicated background daemon thread.

    Decouples asynchronous DDS waitsets and callback dispatch from the synchronous
    simulation/physics stepping loop, preventing dropped callbacks under rendering load.
    """

    def __init__(
        self,
        nodes: Sequence[Node] | None = None,
        executor: Executor | None = None,
    ) -> None:
        """Initialize the background executor.

        Args:
            nodes: Optional sequence of ROS 2 Node instances to attach.
            executor: Optional custom rclpy Executor. If None, a SingleThreadedExecutor is used.
        """
        self._executor = executor
        self._nodes: list[Node] = list(nodes) if nodes is not None else []
        self._thread: threading.Thread | None = None
        self._is_shutdown: bool = False
        self._lock = threading.RLock()

        # If an executor was explicitly passed and nodes were provided, register them
        if self._executor is not None:
            for node in self._nodes:
                self._executor.add_node(node)

    @property
    def executor(self) -> Executor:
        """Get or lazily initialize the underlying rclpy executor."""
        with self._lock:
            if self._executor is None:
                self._executor = SingleThreadedExecutor()
                for node in self._nodes:
                    self._executor.add_node(node)
            return self._executor

    @property
    def is_spinning(self) -> bool:
        """Return True if the background thread is actively running."""
        return self._thread is not None and self._thread.is_alive()

    def add_node(self, node: Node) -> None:
        """Add a ROS 2 node to the executor.

        Args:
            node: The Node instance to add.
        """
        with self._lock:
            if node not in self._nodes:
                self._nodes.append(node)
                if self._executor is not None:
                    self._executor.add_node(node)

    def remove_node(self, node: Node) -> None:
        """Remove a ROS 2 node from the executor.

        Args:
            node: The Node instance to remove.
        """
        with self._lock:
            if node in self._nodes:
                self._nodes.remove(node)
                if self._executor is not None:
                    self._executor.remove_node(node)

    def get_nodes(self) -> list[Node]:
        """Return a copy of the list of nodes managed by this executor."""
        with self._lock:
            return list(self._nodes)

    def start(self) -> None:
        """Start spinning the executor on a dedicated daemon thread."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                _logger.warning("BackgroundRos2Executor is already running.")
                return

            self._is_shutdown = False
            # Ensure executor is instantiated
            _ = self.executor

            self._thread = threading.Thread(
                target=self._spin_worker,
                daemon=True,
                name="BackgroundRos2Executor",
            )
            self._thread.start()

    def _spin_worker(self) -> None:
        """Target loop running inside the background daemon thread."""
        while not self._is_shutdown and rclpy.ok():
            try:
                self.executor.spin_once(timeout_sec=0.05)
            except (ExternalShutdownException, ShutdownException):
                break
            except Exception as e:
                if not self._is_shutdown and rclpy.ok():
                    _logger.error(f"Unexpected error in BackgroundRos2Executor: {e}", exc_info=True)
                break

    def shutdown(self, timeout_sec: float = 2.0) -> None:
        """Safely shut down the executor and join the background worker thread.

        Args:
            timeout_sec: Maximum time in seconds to wait for thread join.
        """
        with self._lock:
            self._is_shutdown = True
            if self._executor is not None:
                try:
                    self._executor.shutdown()
                except Exception:
                    pass

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout_sec)
            if self._thread.is_alive():
                _logger.warning(
                    f"BackgroundRos2Executor thread did not terminate within {timeout_sec}s."
                )

    def __enter__(self) -> BackgroundRos2Executor:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()
