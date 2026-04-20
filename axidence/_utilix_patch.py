"""Back-port of later-utilix behavior: make the RunDB token-login retry count
configurable via `uconfig.get('RunDB', 'rundb_retry', fallback=3)`.

In utilix 0.7.x (the version shipped in the cvmfs `sr0_wimp` / `2022.06.3`
environments) `Token.new_token` hardcodes `for _try in range(3)` with
`time.sleep(10**_try)` — so a login failure against an unreachable RunDB
API server wastes 110 s before we give up. Later utilix (>=0.9) made the
retry count configurable so users who don't have RunDB access (offline /
no-VPN workstations) can set `rundb_retry = 1` in their `.utilixrc` and
fail fast.

The patch is a no-op on newer utilix — `Token.new_token` already reads
the same config key — and on any installation that doesn't import
utilix.rundb successfully.
"""

from __future__ import annotations

import datetime
import json
import logging
import time

logger = logging.getLogger(__name__)


def patch_utilix_rundb_retry():
    try:
        import utilix.rundb as _rundb
        from utilix import uconfig
    except ImportError:
        return

    # Older utilix installs don't expose a version attribute; inspect the
    # source to see whether `new_token` already reads `rundb_retry`.
    import inspect

    try:
        src = inspect.getsource(_rundb.Token.new_token)
    except (OSError, TypeError):
        return

    if "rundb_retry" in src:
        # Already the modern behavior — nothing to do.
        return

    # Build a replacement new_token that mirrors the upstream (utilix >=0.9)
    # logic. Retry count and sleep pattern are now configurable.
    try:
        import requests  # type: ignore
    except ImportError:
        return

    prefix = getattr(_rundb, "PREFIX", None)
    base_headers = getattr(_rundb, "BASE_HEADERS", None)
    NewTokenError = getattr(_rundb, "NewTokenError", RuntimeError)
    if prefix is None or base_headers is None:
        # utilix internals have moved; don't try to patch.
        return

    def new_token(self):
        tk_rundb_api_url = uconfig.get("RunDB", "tk_rundb_api_url", fallback=None)
        if tk_rundb_api_url:
            paths = [tk_rundb_api_url + "/login", prefix + "/login"]
        else:
            paths = [prefix + "/login"]
        username = uconfig.get("RunDB", "rundb_api_user")
        pw = uconfig.get("RunDB", "rundb_api_password")
        data = json.dumps({"username": username, "password": pw})
        logger.debug("Creating a new token: doing API call now")
        n_try = int(uconfig.get("RunDB", "rundb_retry", fallback=3))
        n_try = max(n_try, 1)
        success = False
        response = None
        response_json: dict = {}
        for _try in range(n_try):
            try:
                for path in paths:
                    response = requests.post(path, data=data, headers=base_headers)
                    response_json = json.loads(response.text)
                    success = True
                    break
                if success:
                    break
            except json.decoder.JSONDecodeError:
                if _try < n_try - 1:
                    sleep_for = min(10**_try, 10)
                    logger.info(
                        f"Login attempt #{_try + 1} failed. "
                        f"Sleeping for {sleep_for} seconds and trying again."
                    )
                    time.sleep(sleep_for)
        if not success:
            raise NewTokenError("Error in creating a token.")
        token = response_json.get("access_token", "CALL_FAILED")
        if token == "CALL_FAILED":
            logger.error(
                "API call to create new token failed. Here is the response:\n"
                f"{response.text if response is not None else ''}"
            )
            raise RuntimeError("Creating a new token failed.")
        self.token_string = token
        self.user = username
        self.creation_time = datetime.datetime.now().timestamp()
        self.write()

    new_token._axidence_patched = True  # type: ignore[attr-defined]
    _rundb.Token.new_token = new_token
