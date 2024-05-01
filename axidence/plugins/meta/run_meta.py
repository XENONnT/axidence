import numpy as np
import strax
from strax import ExhaustPlugin


class RunMeta(ExhaustPlugin):
    """Plugin that provides run metadata."""

    __version__ = "0.0.0"
    depends_on = "event_basics"
    provides = "run_meta"
    data_kind = "run_meta"
    save_when = strax.SaveWhen.EXPLICIT

    dtype = strax.time_fields

    def compute(self, events, start, end):
        result = np.zeros(1, dtype=self.dtype)
        result["time"] = start
        result["endtime"] = end
        return result
