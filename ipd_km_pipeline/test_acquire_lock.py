"""Tests for the manifest durability helpers in acquire_corpus.py:
atomic JSON writes and the single-writer acquisition lock."""
import json
import os
import time

import pytest

import acquire_corpus as A


def test_atomic_write_roundtrip_no_tmp_left(tmp_path):
    p = tmp_path / "manifest.json"
    A._write_json_atomic(p, {"pdfs": [{"pmcid": "1"}], "no_pdf": ["2"]})
    assert json.loads(p.read_text())["no_pdf"] == ["2"]
    # no leftover temp files beside the manifest
    assert [f.name for f in tmp_path.iterdir()] == ["manifest.json"]


def test_atomic_write_overwrites_existing(tmp_path):
    p = tmp_path / "m.json"
    p.write_text('{"old": true}')
    A._write_json_atomic(p, {"new": 1})
    assert json.loads(p.read_text()) == {"new": 1}


def test_lock_taken_and_released(tmp_path):
    lock = A._AcquireLock(tmp_path)
    with lock:
        assert lock.path.exists()                       # held
    assert not lock.path.exists()                       # released on exit


def test_lock_refuses_second_holder_when_fresh(tmp_path):
    with A._AcquireLock(tmp_path):
        with pytest.raises(RuntimeError, match="another acquisition"):
            A._AcquireLock(tmp_path).__enter__()        # second taker refused


def test_lock_takes_over_when_stale(tmp_path):
    lock1 = A._AcquireLock(tmp_path)
    lock1.__enter__()
    # simulate an abandoned/zombie lock: backdate the heartbeat past the stale window
    old = time.time() - A._LOCK_STALE_S - 10
    os.utime(lock1.path, (old, old))
    # a new acquisition may now take over (no RuntimeError)
    with A._AcquireLock(tmp_path) as lock2:
        assert lock2.path.exists()


def test_lock_released_on_exception(tmp_path):
    with pytest.raises(ValueError):
        with A._AcquireLock(tmp_path) as lock:
            assert lock.path.exists()
            raise ValueError("boom")
    assert not (tmp_path / ".acquire.lock").exists()    # released despite the error
