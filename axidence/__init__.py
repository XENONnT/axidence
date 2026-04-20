__version__ = "0.5.0+sr0"

# SR0 release: back-port `rundb_retry` support to the old utilix 0.7.x shipped
# in the sr0_wimp / 2022.06.3 envs so login against an unreachable RunDB API
# fails fast instead of wasting 110 s of exponential-backoff sleep. Must run
# before any utilix-backed context is constructed.
from ._utilix_patch import patch_utilix_rundb_retry as _patch_utilix_rundb_retry

_patch_utilix_rundb_retry()
del _patch_utilix_rundb_retry

from . import dtypes  # noqa: E402
from .dtypes import *  # noqa: E402, F401, F403

from .utils import *  # noqa: E402, F401, F403

from .samplers import *  # noqa: E402, F401, F403

from . import plugins  # noqa: E402
from .plugins import *  # noqa: E402, F401, F403

from .context import *  # noqa: E402, F401, F403
