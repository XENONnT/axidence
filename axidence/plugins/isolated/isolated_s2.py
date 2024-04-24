import numpy as np
import strax
from strax import Plugin
import straxen

from ...utils import needed_dtype, copy_dtype


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

    isolated_s2_fields = straxen.URLConfig(
        default=np.dtype(strax.peak_dtype(n_channels=straxen.n_tpc_pmts)).names,
        type=(list, tuple),
        track=True,
        help="Needed fields in isolated S2",
    )

    groups_seen = 0

    def refer_dtype(self):
        provided_fields, union_dtype = needed_dtype(self.deps, self.dependencies_by_kind, set.union)
        return strax.unpack_dtype(union_dtype)

    def infer_dtype(self):
        dtype_reference = self.refer_dtype()
        dtype = copy_dtype(dtype_reference, self.isolated_s2_fields)
        dtype += [
            (("Group number of peaks", "group_number"), np.int64),
        ]
        return dtype

    def compute(self, events, peaks):
        _events = events[events["cut_isolated_s2"]]
        split_peaks = strax.split_by_containment(peaks, _events)
        _peaks = np.hstack(split_peaks)
        if _events["n_peaks"].sum() != len(_peaks):
            raise ValueError(f"Expected {_events['n_peaks'].sum()} peaks, got {len(_peaks)}!")

        all_names = set(peaks.dtype.names) | set(events.dtype.names)
        result = np.empty(_events["n_peaks"].sum(), dtype=self.dtype)
        for n in result.dtype.names:
            if n not in ["group_number"]:
                if n not in all_names:
                    raise ValueError(f"Field {n} not found in peaks or events!")
                if n in peaks.dtype.names:
                    result[n] = _peaks[n]
                if n in events.dtype.names:
                    result[n] = np.repeat(_events[n], _events["n_peaks"])

        result["group_number"] = (
            np.repeat(np.arange(len(_events)), _events["n_peaks"]) + self.groups_seen
        )

        self.groups_seen += len(_events)
        return result
