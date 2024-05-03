import numpy as np
import strax
from strax import Plugin
import straxen

from ...utils import needed_dtype, copy_dtype
from ...dtypes import positioned_peak_dtype, correlation_fields, event_level_fields


class IsolatedS2(Plugin):
    __version__ = "0.0.0"
    depends_on = (
        "cut_isolated_s2",
        "event_basics",
        "event_positions",
        "event_shadow",
        "event_ambience",
        "event_pattern_fit",
        "peaks",
        "peak_basics",
        "peak_positions",
        "peak_proximity",
        "peak_shadow",
        "peak_ambience",
        "peak_se_density",
        "peak_nearest_triggering",
    )
    provides = "isolated_s2"
    data_kind = "isolated_s2"

    isolated_peaks_fields = straxen.URLConfig(
        default=list(np.dtype(positioned_peak_dtype()).names) + correlation_fields,
        type=(list, tuple),
        help="Needed fields in isolated peaks",
    )

    isolated_events_fields = straxen.URLConfig(
        default=event_level_fields,
        type=(list, tuple),
        help="Needed fields in isolated events",
    )

    groups_seen = 0

    def refer_dtype(self, data_kind):
        provided_fields, union_dtype = needed_dtype(
            self.deps, [self.dependencies_by_kind()[data_kind]], set.union
        )
        return strax.unpack_dtype(union_dtype)

    def infer_dtype(self):
        # Note that here we only save the time and endtime of peaks, but not events
        for n in ["time", "endtime", "x", "y"]:
            if n in self.isolated_events_fields:
                raise ValueError(f"{n} is not allowed in isolated_events_fields!")
        dtype = copy_dtype(self.refer_dtype("peaks"), self.isolated_peaks_fields)
        dtype += copy_dtype(self.refer_dtype("events"), self.isolated_events_fields)
        dtype += [
            (("Run id", "run_id"), np.int32),
            (("Group number of peaks", "group_number"), np.int64),
        ]
        return dtype

    def compute(self, events, peaks):
        _events = events[events["cut_isolated_s2"]]
        if len(_events) == 0:
            return np.empty(0, dtype=self.dtype)

        split_peaks = strax.split_by_containment(peaks, _events)
        _peaks = np.hstack(split_peaks)
        if _events["n_peaks"].sum() != len(_peaks):
            raise ValueError(f"Expected {_events['n_peaks'].sum()} peaks, got {len(_peaks)}!")

        result = np.empty(_events["n_peaks"].sum(), dtype=self.dtype)
        for n in result.dtype.names:
            if n not in ["run_id", "group_number"]:
                if n in self.isolated_peaks_fields:
                    result[n] = _peaks[n]
                elif n in self.isolated_events_fields:
                    result[n] = np.repeat(_events[n], _events["n_peaks"])
                else:
                    raise ValueError(f"Field {n} not found in peaks or events!")

        result["run_id"] = int(self.run_id)
        result["group_number"] = (
            np.repeat(np.arange(len(_events)), _events["n_peaks"]) + self.groups_seen
        )

        self.groups_seen += len(_events)
        return result
