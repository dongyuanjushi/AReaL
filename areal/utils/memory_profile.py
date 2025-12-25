import os
import torch
import contextlib

MEMORY_MAX_ENTRIES = 100000
STORE_BASE_DIR = "/storage/openpsi/users/meizhiyu.mzy/zeta/memory_profile"

PROFILE_MEMORY = os.environ.get("PROFILE_MEMORY", "0") == "1"

@contextlib.contextmanager
def profile_memory(fn="memory.pickle"):
    if not PROFILE_MEMORY:
        yield
        return
    torch.cuda.memory._record_memory_history(max_entries=MEMORY_MAX_ENTRIES)
    yield
    fp = os.path.join(STORE_BASE_DIR, fn)
    print(f"Dumping memory profile to {fp}")
    torch.cuda.memory._dump_snapshot(fp)
    torch.cuda.memory._record_memory_history(enabled=None)