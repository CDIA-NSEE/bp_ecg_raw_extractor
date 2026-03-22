"""In-memory deduplication store for bp_ecg_raw_extractor.

Tracks S3 object keys that have already been processed in the current
process lifetime.  Uses a thread-safe ``set`` so the polling loop and
concurrent ``asyncio.gather`` calls never reprocess the same object.

Note: this store is ephemeral — it resets when the process restarts.
For persistent deduplication, replace with a DedupStore backed by SQLite
or Redis in a future iteration.
"""

from __future__ import annotations

import threading


class ProcessedKeyStore:
    """Thread-safe in-memory set of already-processed S3 object keys."""

    def __init__(self) -> None:
        self._processed: set[str] = set()
        self._lock: threading.Lock = threading.Lock()

    def is_processed(self, key: str) -> bool:
        """Return ``True`` if *key* has already been processed.

        Args:
            key: S3 object key to check.
        """
        with self._lock:
            return key in self._processed

    def mark_processed(self, key: str) -> None:
        """Record *key* as processed.

        Args:
            key: S3 object key to mark.
        """
        with self._lock:
            self._processed.add(key)

    def reset(self) -> None:
        """Clear all recorded keys (useful in tests)."""
        with self._lock:
            self._processed.clear()
