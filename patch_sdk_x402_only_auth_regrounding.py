#!/usr/bin/env python3
"""Regrounds sdk.py/sdk.js/README.md/openapi.json against the real server auth
state after repo commit 4e09c52 ("fix(x402): drop the X-API-Key gate on the 3
real paid routes/tools", 2026-08-25): search()/compute_calibrated_alpha()/
score_pair() (and their JS equivalents) no longer need an API key on the real
server -- x402 payment alone is sufficient, and Stripe metered billing
explicitly excludes these 3 routes (_NEXUS_BILLING_EXCLUDED_PATHS in
core/similarity_search_api_api.py). The SDK still required a non-empty
api_key to even construct a Client, and README.md/openapi.json still
documented/declared X-API-Key as mandatory on those routes -- real friction
for an external caller who only has a wallet, no key. The deprecated,
unpriced, unversioned /similarity/calibrate-alpha stub keeps its gate
untouched (per 4e09c52's own commit message), so openapi.json's security
entry for that one path is intentionally left in place.

Convention per CLAUDE.md SS4: backup, ast.parse()/node --check before+after,
exact block match (count==1) or abort, idempotency marker, verify on disk.
"""
import ast
import json
import shutil
import subprocess
import sys

ROOT = "."

PY_PATH = f"{ROOT}/sdk_wrappers/similarity_search_api_sdk.py"
JS_PATH = f"{ROOT}/sdk_wrappers/sdk.js"
README_PATH = f"{ROOT}/README.md"
OPENAPI_PATH = f"{ROOT}/openapi.json"

MARKER = "sdk_x402_only_auth_regrounding"

PY_OLD = '''class Client:
    def __init__(
        self,
        api_key: str,
        base_url: str = SIMILARITY_SEARCH_BASE_URL,
        timeout: float = SIMILARITY_SEARCH_DEFAULT_TIMEOUT,
        max_retries: int = SIMILARITY_SEARCH_MAX_RETRIES,
    ):
        if not api_key or not isinstance(api_key, str):
            raise SimilaritySearchAuthError(
                "A non-empty 'api_key' string is required to initialize the Client"
            )
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._http = httpx.Client(
            headers={
                # --- PATCH sdk_route_grounding_manual_backfill ---
                # El server real exige X-API-Key (APIKeyHeader), no
                # Authorization: Bearer -- ver core/similarity_search_api_api.py.
                "X-API-Key": self._api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "similarity-search-sdk-python/1.0.0",
            },
            timeout=self._timeout,
        )'''

PY_NEW = '''class Client:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = SIMILARITY_SEARCH_BASE_URL,
        timeout: float = SIMILARITY_SEARCH_DEFAULT_TIMEOUT,
        max_retries: int = SIMILARITY_SEARCH_MAX_RETRIES,
    ):
        # --- PATCH sdk_x402_only_auth_regrounding ---
        # api_key is optional as of repo commit 4e09c52 (2026-08-25): the
        # server dropped the X-API-Key gate on search()/compute_calibrated_alpha()/
        # score_pair(), x402 payment alone is sufficient now, and Stripe metered
        # billing explicitly excludes these 3 routes
        # (_NEXUS_BILLING_EXCLUDED_PATHS in core/similarity_search_api_api.py).
        # Still sent when the caller does pass one, for forward-compat with any
        # future re-gating (e.g. the deprecated /similarity/calibrate-alpha stub,
        # which kept its gate).
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "similarity-search-sdk-python/1.0.0",
        }
        if api_key:
            headers["X-API-Key"] = api_key
        self._http = httpx.Client(headers=headers, timeout=self._timeout)'''

JS_OLD = '''function resolveApiKey(options) {
  const key = (options && options.apiKey) || process.env.SIMILARITY_API_KEY;
  if (!key || typeof key !== 'string' || key.trim().length === 0) {
    throw new AuthenticationError();
  }
  return key.trim();
}

function buildAxiosInstance(apiKey, timeoutMs) {
  return axios.create({
    baseURL: BASE_URL,
    timeout: timeoutMs || DEFAULT_TIMEOUT_MS,
    headers: {
      // --- PATCH sdk_route_grounding_manual_backfill ---
      // El server real exige X-API-Key (APIKeyHeader), no
      // Authorization: Bearer -- ver core/similarity_search_api_api.py.
      'X-API-Key': apiKey,
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      'X-Client': 'similarity-search-sdk-js/1.0.0'
    }
  });
}'''

JS_NEW = '''function resolveApiKey(options) {
  // --- PATCH sdk_x402_only_auth_regrounding ---
  // Optional as of repo commit 4e09c52 (2026-08-25): the server dropped the
  // X-API-Key gate on search()/computeCalibratedAlpha()/scorePair(), x402
  // payment alone is sufficient now, and Stripe metered billing explicitly
  // excludes these 3 routes (_NEXUS_BILLING_EXCLUDED_PATHS in
  // core/similarity_search_api_api.py). No longer throws when absent.
  const key = (options && options.apiKey) || process.env.SIMILARITY_API_KEY;
  if (!key || typeof key !== 'string' || key.trim().length === 0) {
    return null;
  }
  return key.trim();
}

function buildAxiosInstance(apiKey, timeoutMs) {
  const headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'X-Client': 'similarity-search-sdk-js/1.0.0'
  };
  if (apiKey) {
    // Still sent when the caller does pass one, for forward-compat with any
    // future re-gating (e.g. the deprecated /similarity/calibrate-alpha stub,
    // which kept its gate) -- see resolveApiKey() above.
    headers['X-API-Key'] = apiKey;
  }
  return axios.create({
    baseURL: BASE_URL,
    timeout: timeoutMs || DEFAULT_TIMEOUT_MS,
    headers
  });
}'''

README_OLD = """## Authentication

All business endpoints require an `X-API-Key` header:
X-API-Key: <your key>

`/health` requires no authentication."""

README_NEW = """## Authentication

<!-- PATCH sdk_x402_only_auth_regrounding -->
The 3 business endpoints (`/similarity/search`, `/similarity/calibrate-alpha/v1`,
`/similarity/batch-score`) require **no API key** -- only a valid x402 payment
(see "Pricing" below). The `X-API-Key` gate was dropped from these routes
2026-08-25; Stripe metered billing explicitly excludes them too, so passing a
key does nothing on these 3 routes today. The deprecated, unpriced
`/similarity/calibrate-alpha` (no `/v1` suffix) still requires `X-API-Key` and
always 501s regardless.

`/health` requires no authentication."""

# The 3 real paid routes that lost their gate 2026-08-25 -- NOT the deprecated
# unversioned /similarity/calibrate-alpha, which keeps its APIKeyHeader security
# entry untouched (matches the live server / commit 4e09c52).
OPENAPI_DEGATE = [
    ("/similarity/search", "post"),
    ("/similarity/calibrate-alpha/v1", "post"),
    ("/similarity/batch-score", "post"),
]


def patch_py():
    with open(PY_PATH, encoding="utf-8") as f:
        original = f.read()
    ast.parse(original)
    if f"# --- PATCH {MARKER} ---" in original or "api_key: str | None = None" in original:
        print(f"[SKIP] {PY_PATH} already patched")
        return
    if original.count(PY_OLD) != 1:
        print(f"[ABORT] {PY_PATH}: expected exactly 1 match for PY_OLD, found {original.count(PY_OLD)}")
        sys.exit(1)
    shutil.copy2(PY_PATH, PY_PATH + ".bak2")
    patched = original.replace(PY_OLD, PY_NEW)
    ast.parse(patched)
    with open(PY_PATH, "w", encoding="utf-8") as f:
        f.write(patched)
    assert f"# --- PATCH {MARKER} ---" in patched
    print(f"[OK] {PY_PATH} patched")


def patch_js():
    with open(JS_PATH, encoding="utf-8") as f:
        original = f.read()
    if f"// --- PATCH {MARKER} ---" in original:
        print(f"[SKIP] {JS_PATH} already patched")
        return
    if original.count(JS_OLD) != 1:
        print(f"[ABORT] {JS_PATH}: expected exactly 1 match for JS_OLD, found {original.count(JS_OLD)}")
        sys.exit(1)
    shutil.copy2(JS_PATH, JS_PATH + ".bak2")
    patched = original.replace(JS_OLD, JS_NEW)
    with open(JS_PATH, "w", encoding="utf-8") as f:
        f.write(patched)
    result = subprocess.run(["node", "--check", JS_PATH], capture_output=True, text=True)
    if result.returncode != 0:
        shutil.copy2(JS_PATH + ".bak2", JS_PATH)
        print(f"[ABORT] {JS_PATH}: node --check failed after patch:\\n{result.stderr}")
        sys.exit(1)
    assert f"// --- PATCH {MARKER} ---" in patched
    print(f"[OK] {JS_PATH} patched")


def patch_readme():
    with open(README_PATH, encoding="utf-8-sig") as f:
        original = f.read()
    if f"<!-- PATCH {MARKER} -->" in original:
        print(f"[SKIP] {README_PATH} already patched")
        return
    if original.count(README_OLD) != 1:
        print(f"[ABORT] {README_PATH}: expected exactly 1 match for README_OLD, found {original.count(README_OLD)}")
        sys.exit(1)
    shutil.copy2(README_PATH, README_PATH + ".bak2")
    patched = original.replace(README_OLD, README_NEW)
    with open(README_PATH, "w", encoding="utf-8-sig") as f:
        f.write(patched)
    assert f"<!-- PATCH {MARKER} -->" in patched
    print(f"[OK] {README_PATH} patched")


def patch_openapi():
    with open(OPENAPI_PATH, encoding="utf-8") as f:
        original_text = f.read()
    data = json.loads(original_text)  # validate before

    already_clean = all(
        "security" not in data["paths"][path][method]
        for path, method in OPENAPI_DEGATE
    )
    if already_clean:
        print(f"[SKIP] {OPENAPI_PATH} already patched")
        return

    shutil.copy2(OPENAPI_PATH, OPENAPI_PATH + ".bak2")
    changed = 0
    for path, method in OPENAPI_DEGATE:
        op = data["paths"][path][method]
        if "security" in op:
            del op["security"]
            changed += 1
    if changed != 3:
        print(f"[ABORT] {OPENAPI_PATH}: expected to remove 'security' from 3 operations, removed {changed}")
        sys.exit(1)

    # Deprecated unversioned /similarity/calibrate-alpha must still be gated.
    deprecated_op = data["paths"]["/similarity/calibrate-alpha"]["post"]
    assert deprecated_op.get("security") == [{"APIKeyHeader": []}], (
        "deprecated /similarity/calibrate-alpha lost its security entry -- aborting, this route must stay gated"
    )

    patched_text = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    json.loads(patched_text)  # validate after
    with open(OPENAPI_PATH, "w", encoding="utf-8") as f:
        f.write(patched_text)
    print(f"[OK] {OPENAPI_PATH} patched (removed security from {changed} operations)")


if __name__ == "__main__":
    patch_py()
    patch_js()
    patch_readme()
    patch_openapi()
    print("Done.")
