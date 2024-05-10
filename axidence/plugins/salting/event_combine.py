import strax
import numpy as np
from strax import ExhaustPlugin


class EventsCombine(ExhaustPlugin):
    __version__ = "0.0.0"
    depends_on = (
        "events_salting",
        "event_basics_salted",
        "event_shadow_salted",
        "event_ambience_salted",
        "event_nearest_triggering_salted",
        "event_se_density_salted",
    )
    provides = "events_combine"
    data_kind = "events_salted"
    save_when = strax.SaveWhen.EXPLICIT

    def infer_dtype(self):
        dtype = [
            (("Original time of salting events", "origin_time"), np.int64),
        ]
        dtype += strax.merged_dtype(
            [
                self.deps[d].dtype_for(d)
                # Sorting is needed here to match what strax.Chunk does in merging
                for d in sorted(self.depends_on)
            ]
        )
        return dtype

    def compute(self, events_salting, events_salted):
        if len(events_salting) != len(events_salted):
            raise ValueError("Length of salting and salted events do not match")

        result = np.empty(len(events_salting), dtype=self.dtype)

        for n in events_salted.dtype.names:
            result[n] = events_salted[n]

        for n in events_salting.dtype.names:
            if n not in "time endtime".split(" "):
                result[n] = events_salting[n]

        result["origin_time"] = events_salting["time"]

        return result
