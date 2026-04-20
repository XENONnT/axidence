"""Top-level pytest conftest.

The SR0 utilix 0.7.x hardcodes a 3-attempt exponential-backoff token login
(0 + 10 + 100 = 110 s wait on failure), and that runs at `import straxen`
time via admix. We patch `utilix.rundb.Token.new_token` *before* anything
imports straxen/admix so the retry count falls back to the user's utilixrc
`RunDB.rundb_retry` value. conftest.py is evaluated before test modules,
which is early enough.
"""

from axidence._utilix_patch import patch_utilix_rundb_retry as _patch

_patch()
del _patch

import shutil  # noqa: E402

import pytest  # noqa: E402


@pytest.fixture(scope="module")
def rm_strax_data():
    """Remove ./strax_test_data directory before and after initializing the
    TestCase."""
    shutil.rmtree("./strax_test_data", ignore_errors=True)
    yield
    shutil.rmtree("./strax_test_data", ignore_errors=True)
