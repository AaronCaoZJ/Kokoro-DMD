#!/usr/bin/env python3
"""Host-RAM watchdog for Kokoro DMD2 training.

Why this exists
---------------
The DataLoader side (WDS shuffle buffer x num_workers x 2 loaders x 8 ranks,
holding fully .decode("pil")'d i2i story samples at ~200 MB each) can grow host
RAM until the container's cgroup memory.max (e.g. ~1.76 TiB) is hit. When that
happens the kernel OOM-killer fires indiscriminately and can wedge the whole
container -- SSH can no longer even fork. This watchdog trips at a limit that
sits *below* memory.max and SIGKILLs only the training process group, so the job
fails with a clear log line instead of taking the container down with it.

What it watches
---------------
`anon` from /sys/fs/cgroup/memory.stat -- the *unreclaimable* anonymous memory,
which is what the shuffle buffers actually consume and what drives a true OOM.
It deliberately does NOT watch memory.current, because that includes reclaimable
page cache from streaming the multi-GB WDS tar shards; the kernel drops that
under pressure, so keying on it would cause false trips.

Target selection
----------------
GUARD_PGID_FILE points at a file containing the process-group id of the training
job (written by the launch script's setsid'd subshell, whose $$ == pgid). The
whole torchrun rank+worker tree shares that pgid; the launcher and this watchdog
live in different groups, so `killpg` hits only training.

Env knobs
---------
  GUARD_PGID_FILE   path to file holding the training pgid (required)
  GUARD_LIMIT_GIB   trip threshold in GiB (default 1450; also capped at 90% of
                    the cgroup's memory.max if that is readable)
  GUARD_POLL_SEC    poll interval seconds (default 3)
  GUARD_GRACE_SEC   seconds between SIGTERM and SIGKILL (default 5)
  GUARD_SENTINEL    path to write a trip marker (default /tmp/mem_guard_tripped)
"""
import os
import signal
import sys
import time

CGROUP = "/sys/fs/cgroup"
GIB = 1024 ** 3


def read_anon_bytes():
    with open(f"{CGROUP}/memory.stat") as f:
        for line in f:
            if line.startswith("anon "):
                return int(line.split()[1])
    return 0


def read_pod_max_bytes():
    try:
        v = open(f"{CGROUP}/memory.max").read().strip()
        return None if v == "max" else int(v)
    except Exception:
        return None


def read_pgid(path):
    try:
        s = open(path).read().strip()
        return int(s) if s else None
    except Exception:
        return None


def log(msg):
    line = f"[mem_guard] {msg}"
    print(line, flush=True)
    sys.stderr.write(line + "\n")
    sys.stderr.flush()


def main():
    pgid_file = os.environ.get("GUARD_PGID_FILE")
    if not pgid_file:
        log("FATAL: GUARD_PGID_FILE not set; refusing to run blind")
        return 2

    limit = int(float(os.environ.get("GUARD_LIMIT_GIB", "1450")) * GIB)
    pod_max = read_pod_max_bytes()
    if pod_max:
        limit = min(limit, int(pod_max * 0.90))
    poll = float(os.environ.get("GUARD_POLL_SEC", "3"))
    grace = float(os.environ.get("GUARD_GRACE_SEC", "5"))
    sentinel = os.environ.get("GUARD_SENTINEL", "/tmp/mem_guard_tripped")

    # Wait for the launcher to publish the training pgid.
    pgid = None
    for _ in range(100):
        pgid = read_pgid(pgid_file)
        if pgid:
            break
        time.sleep(0.1)
    if not pgid:
        log(f"FATAL: no pgid in {pgid_file} after 10s; exiting")
        return 2

    log(
        f"armed: pgid={pgid} limit={limit/GIB:.0f}GiB "
        f"pod_max={(pod_max or 0)/GIB:.0f}GiB poll={poll}s grace={grace}s"
    )

    peak = 0
    while True:
        # Stop quietly once the training group is gone.
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            log(f"training pgid={pgid} exited; watchdog stopping (peak anon={peak/GIB:.1f}GiB)")
            return 0
        except PermissionError:
            pass  # group exists but not signalable by us; keep watching

        anon = read_anon_bytes()
        if anon > peak:
            peak = anon

        if anon >= limit:
            log(
                f"FATAL: anon={anon/GIB:.1f}GiB >= limit={limit/GIB:.0f}GiB "
                f"(pod_max={(pod_max or 0)/GIB:.0f}GiB). Killing training pgid={pgid} "
                f"to spare the pod."
            )
            try:
                with open(sentinel, "w") as f:
                    f.write(f"tripped at anon={anon/GIB:.1f}GiB limit={limit/GIB:.0f}GiB\n")
            except Exception:
                pass
            try:
                os.killpg(pgid, signal.SIGTERM)
            except Exception as e:
                log(f"SIGTERM killpg failed: {e}")
            # Under memory pressure handlers may never run; escalate to SIGKILL.
            deadline = time.time() + grace
            while time.time() < deadline:
                try:
                    os.killpg(pgid, 0)
                except ProcessLookupError:
                    log("training group exited after SIGTERM")
                    return 137
                time.sleep(0.25)
            try:
                os.killpg(pgid, signal.SIGKILL)
                log("escalated to SIGKILL")
            except Exception as e:
                log(f"SIGKILL killpg failed: {e}")
            return 137

        time.sleep(poll)


if __name__ == "__main__":
    sys.exit(main())
