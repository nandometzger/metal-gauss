"""The kernel build's lock file, checked without building anything.

torch waits on `<build_dir>/lock` with a bare `while os.path.exists(lock):
sleep(0.1)` and prints nothing, so a lock left behind by an interrupted build
makes every later run sit forever with no output. That looks exactly like a
hung GPU, which is the wrong thing to go debugging: one render here sat for
five minutes before a `sample` showed it asleep in `time_sleep`.

Nothing is deleted automatically -- two processes may legitimately build at
once -- so all of this is a message, and a message is testable on any machine.
"""

from __future__ import annotations

import os

from metal_gauss.metal_backend import stale_lock_message


def _lock(dir_path, age_s, now=1_000_000.0):
    lock = dir_path / "lock"
    lock.touch()
    os.utime(lock, (now - age_s, now - age_s))
    return lock


def test_no_lock_means_nothing_to_say(tmp_path):
    assert stale_lock_message(tmp_path, now=1_000_000.0) is None


def test_a_missing_build_directory_says_nothing(tmp_path):
    assert stale_lock_message(tmp_path / "never-built", now=1_000_000.0) is None


def test_a_fresh_lock_reports_another_build(tmp_path):
    """Seconds old: someone else is very likely compiling right now."""
    lock = _lock(tmp_path, age_s=5.0)
    message = stale_lock_message(tmp_path, now=1_000_000.0)

    assert message is not None
    assert str(lock) in message
    assert "5s" in message
    assert "stale" not in message.lower(), "a five-second-old lock is not stale"


def test_an_old_lock_is_named_as_stale_and_says_what_to_do(tmp_path):
    """Older than any build takes, so the wait will never end on its own."""
    lock = _lock(tmp_path, age_s=3_600.0)
    message = stale_lock_message(tmp_path, now=1_000_000.0)

    assert message is not None
    assert str(lock) in message
    assert "stale" in message.lower()
    assert "delete" in message.lower() or "remove" in message.lower()
