# Localhost Security Hardening — Design

- **Date:** 2026-06-08
- **Branch:** feature/pyinstaller-packaging
- **Status:** Approved (design) — pending implementation plan
- **Scope decision:** Local-only deployment, Approach B (minimal fix + cheap hardening), with tests.

## Problem

A code review of the branch surfaced one Crucial and one Important security issue in `app.py`:

1. **Crucial — unauthenticated API exposed to the network.** The FastAPI server binds `host="0.0.0.0"` and applies CORS `allow_origins=["*"]` / `allow_credentials=True`, with no authentication. Every endpoint (delete shows/profiles, drive the physical DMX hardware, trigger LLM generation that spends cloud money) is reachable by any device on the same LAN/Wi-Fi.
2. **Important — path traversal in `_resolve_show_path`.** The guard uses `realpath(show_dir).startswith(realpath(SHOWS_DIR))`. Because `shows` is a string prefix of `shows_backup`, `show_id="../shows_backup"` passes the check, and `DELETE /api/shows/{id}` → `shutil.rmtree` can delete sibling `shows*` directories.

**Confirmed usage model:** the app is used *locally only* — the browser/UI always runs on the same machine as the backend. No phone/LAN control is intended, so **no authentication is required**.

## Goals

- Stop the API from being reachable by other machines on the network.
- Close the residual "a malicious website open in the user's browser POSTs to `localhost:8000`" vector (CSRF-style + DNS rebinding), which localhost-binding alone does *not* fix.
- Fix the path-traversal prefix bug correctly.
- Add the repo's first automated tests to lock these behaviors in.

## Non-Goals

- No authentication / token (unnecessary for a same-machine tool).
- No changes to the DMX engines, LLM provider code, PWA, or packaging.
- No fix for the (separate) engine code-duplication or Bedrock-truncation items from the review — those are tracked elsewhere.

## Design

All changes are in `app.py`.

### Change 1 — Bind to the loopback interface (the Crucial fix)
In `__main__`, change `uvicorn.run(app, host="0.0.0.0", port=8000)` to `host="127.0.0.1"`. Update the frozen-mode auto-open to `http://127.0.0.1:8000` so it matches the bind exactly (avoids a `localhost`→IPv6 `::1` resolution mismatch where the browser hits `::1` but the server only listens on IPv4 `127.0.0.1`).

Safe because: prod serves the UI same-origin (relative URLs), and dev uses the Vite proxy which targets `:8000` server-side — nothing depends on the wildcard bind.

### Change 2 — Fix the path-traversal check (Important)
Replace the `startswith` guard with a component-aware `commonpath` check:

```python
def _resolve_show_path(show_id: str) -> str:
    base = os.path.realpath(SHOWS_DIR)
    show_dir = os.path.realpath(os.path.join(base, show_id))
    try:
        if os.path.commonpath([base, show_dir]) != base:
            raise HTTPException(400, "Invalid show ID")
    except ValueError:          # different drive / malformed path (Windows)
        raise HTTPException(400, "Invalid show ID")
    return show_dir
```

`commonpath` compares whole path components, so `shows_backup` no longer matches the `shows` prefix. The `ValueError` guard covers the Windows case where `show_id` is an absolute path on a different drive (e.g., `E:\evil`), which makes `commonpath` raise.

### Change 3 — Tighten CORS
Replace the wildcard config with an explicit localhost allowlist and drop credentials (no cookies are used):

```python
allow_origins=[
    "http://localhost:5173", "http://127.0.0.1:5173",   # Vite dev
    "http://localhost:8000", "http://127.0.0.1:8000",   # packaged / same-origin
],
allow_credentials=False,
allow_methods=["*"],
allow_headers=["*"],
```

A malicious external origin's CORS **preflight** (both JSON `POST` and `DELETE` trigger one) now fails, blocking the request before it executes. No effect on dev (Vite proxy = same-origin) or prod (same-origin).

### Change 4 — DNS-rebinding defense
Add Starlette's built-in `TrustedHostMiddleware`:

```python
from starlette.middleware.trustedhost import TrustedHostMiddleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1"])
```

A rebinding domain (`evil.com` → `127.0.0.1`) sends `Host: evil.com` and is rejected with `400`.

## Testing Plan

Add `pytest` + FastAPI `TestClient` tests (no hardware required). This is the repo's first test infra.

- `test_resolve_show_path`: rejects `../shows_backup`, `..`, absolute paths, and (Windows) a different-drive path; accepts a normal id.
- `test_cors_preflight_rejects_foreign_origin`: an `OPTIONS` with `Origin: https://evil.com` does **not** receive an `access-control-allow-origin` echo for that origin.
- `test_trusted_host_rejects_bad_host`: a request with `Host: evil.com` returns `400`.

## Implementation Notes / Edge Cases to Confirm

- **`TrustedHostMiddleware` + `TestClient`:** the test client defaults to `Host: testserver`, which the allowlist rejects. Construct the client as `TestClient(app, base_url="http://localhost:8000")` rather than adding `testserver` to the production allowlist.
- **`TrustedHostMiddleware` port handling [Inferred]:** confirm it matches the host ignoring the `:8000` port (i.e., `localhost:8000` matches `localhost`). If it does not, include port-qualified entries.
- **Middleware order:** add `TrustedHostMiddleware` so it runs early (rejecting bad hosts before routing); verify it composes correctly with the CORS middleware's `OPTIONS` handling.
- **Bind vs. workers:** only `app.py` opens a socket; the engine/analyzer workers use file IPC, so no other binding changes are needed.

## Rollback

Each change is independent and revertible in isolation. If the loopback bind breaks an unforeseen local workflow, reverting Change 1 alone restores prior behavior. The CORS/TrustedHost changes can be reverted independently if a legitimate local origin/host is unexpectedly blocked.

## Risks & Assumptions

1. **Assumption: truly local-only.** If phone/LAN control is ever wanted later, the loopback bind must be revisited *together with* adding authentication — do not simply re-expose `0.0.0.0` without auth.
2. **Assumption: the path-traversal endpoints are the only `rmtree`/file sinks fed by request input.** The fix hardens `_resolve_show_path`; if other endpoints build `shows/`-relative paths without it, they need the same treatment (to verify during implementation).
