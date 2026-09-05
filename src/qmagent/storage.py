"""Atomic JSON commits + advisory single-writer lock; no pickle deserialization.

The journal is authoritative. Each record commits observations AND simulator RNG
together. A crash before commit can safely replay only because this is a simulator.
Hashes detect accidental corruption, not malicious rewriting by the file owner.
"""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


def encoded(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def digest(value) -> str:
    return hashlib.sha256(encoded(value)).hexdigest()


def read_json(path: Path):
    if path.stat().st_size > 32 * 1024**2:
        raise ValueError(f"JSON record too large: {path.name}")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate key: {key}")
            result[key] = value
        return result
    def constant(value):
        raise ValueError(f"Nonfinite JSON: {value}")
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique, parse_constant=constant)


def atomic_json(path: Path, value, replace=False):
    payload = encoded(value)
    fd, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)  # Atomic create; never overwrite a committed record.
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


class SessionLock:
    def __init__(self, directory: Path):
        self.handle = (directory / ".lock").open("a+")
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.handle.close()
            raise ValueError("Session is already locked by another terminal; concurrent changes are not allowed") from exc

    def close(self):
        if not self.handle.closed:
            fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()


class Journal:
    def __init__(self, directory: Path, create=False):
        self.directory = directory / "journal"
        if create:
            self.directory.mkdir(mode=0o700, exist_ok=True)
        self.sequence, self.previous = 0, None

    def records(self):
        previous = None
        for sequence, path in enumerate(sorted(self.directory.glob("*.json"))):
            if path.name != f"{sequence:06d}.json":
                raise ValueError("Journal gap/invalid filename; refusing partial resume")
            record = read_json(path)
            body = record["body"]
            if (body["sequence"] != sequence or body["previous"] != previous
                    or record["sha256"] != digest(body)):
                raise ValueError("Journal checksum/chain mismatch")
            previous = record["sha256"]
            yield body, previous

    def recover(self):
        latest = None
        for body, checksum in self.records():
            latest = body
            self.sequence, self.previous = body["sequence"] + 1, checksum
        if latest is None:
            raise ValueError("Session has no committed initial state")
        return latest

    def append(self, snapshot, events, metadata_sha256):
        body = {"sequence": self.sequence, "previous": self.previous,
                "metadata_sha256": metadata_sha256, "snapshot": snapshot, "events": events}
        checksum = digest(body)
        atomic_json(self.directory / f"{self.sequence:06d}.json", {"body": body, "sha256": checksum})
        self.sequence += 1
        self.previous = checksum


def session_directory(home: Path, session_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", session_id):
        raise ValueError("Use a session ID, not a path")
    root = (home / "sessions").resolve()
    directory = root / session_id
    if directory.is_symlink():
        raise ValueError("Symlinked sessions are not supported")
    return directory
