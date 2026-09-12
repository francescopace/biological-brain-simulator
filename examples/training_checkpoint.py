"""Immutable sample-boundary checkpoints for reproducible training experiments."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from src.persistence import load_brain, save_brain


def array_digest(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def directory_digest(path):
    digest = hashlib.sha256()
    for file in sorted(Path(path).rglob("*")):
        if file.is_file():
            digest.update(str(file.relative_to(path)).encode())
            with file.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{path.name}-", dir=path.parent) as folder:
        staged = Path(folder) / "data.json"
        staged.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        os.replace(staged, path)


def save_training_checkpoint(brain, path, progress):
    """Publish brain and cursor together; never replace an existing checkpoint.

    Directory rename publishes a fully written sample boundary. As with the
    underlying brain serializer, this does not promise durability on power loss.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Checkpoint already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{path.name}-", dir=path.parent) as folder:
        staged = Path(folder)
        save_brain(brain, staged / "brain")
        metadata = {
            "version": 1, "progress": progress,
            "brain_step_count": brain.step_count, "brain_time": brain.time,
            "brain_sha256": directory_digest(staged / "brain"),
        }
        (staged / "progress.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        if path.exists():
            raise FileExistsError(f"Checkpoint already exists: {path}")
        os.rename(staged, path)


def load_training_checkpoint(path, expected_protocol):
    path = Path(path)
    metadata = json.loads((path / "progress.json").read_text())
    if metadata["version"] != 1:
        raise ValueError("Unsupported training checkpoint version")
    progress = metadata["progress"]
    if progress["protocol"] != expected_protocol:
        raise ValueError("Training checkpoint protocol/data mismatch")
    if directory_digest(path / "brain") != metadata["brain_sha256"]:
        raise ValueError("Training checkpoint integrity check failed")
    brain = load_brain(path / "brain")
    if brain.step_count != metadata["brain_step_count"] or brain.time != metadata["brain_time"]:
        raise ValueError("Training checkpoint cursor/state mismatch")
    return brain, progress
