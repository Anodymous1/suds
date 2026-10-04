import os
import sys
import threading
import time
import psutil
def _tree_memory(proc, use_pss=False):
    """Total memory of proc + all descendants. Returns (bytes, n_children, biggest_child)."""
    total, biggest = 0, (None, 0)
    children = proc.children(recursive=True)
    for p in [proc] + children:
        try:
            if use_pss:
                m = p.memory_full_info().pss   # Linux only, slower, avoids double-counting
            else:
                m = p.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        total += m
        if p is not proc and m > biggest[1]:
            biggest = (p.pid, m)
    return total, len(children), biggest


def _kill_tree_and_exit(proc, code=1):
    children = proc.children(recursive=True)
    for c in children:
        try:
            c.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(children, timeout=5)
    os._exit(code)   # hard exit; skips joblib cleanup that could hang


def start_memory_watchdog(limit_gb, min_available_gb=None,
                          interval=2.0, log_every=15.0, use_pss=False):
    """
    limit_gb:         kill if this program (main + workers) exceeds this
    min_available_gb: also kill if the whole machine drops below this much free RAM
    """
    main = psutil.Process(os.getpid())
    limit = limit_gb * 1024**3
    GB = 1024**3

    def run():
        peak, last_log = 0, 0.0
        while True:
            used, n_children, (big_pid, big_mem) = _tree_memory(main, use_pss)
            peak = max(peak, used)
            avail = psutil.virtual_memory().available
            now = time.time()

            if now - last_log >= log_every:
                print(f"[mem {time.strftime('%H:%M:%S')}] "
                      f"total={used/GB:.2f} GB  peak={peak/GB:.2f} GB  "
                      f"workers={n_children}  biggest=pid {big_pid} ({big_mem/GB:.2f} GB)  "
                      f"system_free={avail/GB:.1f} GB",
                      flush=True)
                last_log = now

            reason = None
            if used > limit:
                reason = f"program using {used/GB:.2f} GB > limit {limit_gb} GB"
            elif min_available_gb is not None and avail < min_available_gb * GB:
                reason = f"system free RAM {avail/GB:.2f} GB < {min_available_gb} GB"

            if reason:
                print(f"\n[mem] KILLING: {reason}", file=sys.stderr, flush=True)
                _kill_tree_and_exit(main)

            time.sleep(interval)

    t = threading.Thread(target=run, daemon=True, name="mem-watchdog")
    t.start()
    return t