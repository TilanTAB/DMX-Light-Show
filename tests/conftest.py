"""Suite-wide setup. DMX_DRY_RUN must be set BEFORE any test imports
dmx_engine, because the module caches DRY_RUN at import time. Previously only
two test files set it at module scope, so the suite's dry-run state depended
on alphabetical collection order and standalone file runs silently ran with
DRY_RUN=False. conftest.py imports before any test module, closing that gap."""
import os

os.environ["DMX_DRY_RUN"] = "1"
