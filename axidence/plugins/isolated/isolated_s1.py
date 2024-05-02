import numpy as np
import strax
from strax import Plugin
import straxen

from ...utils import needed_dtype, copy_dtype
from ...dtypes import positioned_peak_dtype, correlation_fields


class IsolatedS1(Plugin):
    __version__ = "0.0.0"
    depends_on = (
        "cut_isolated_s1",
        "peaks",
        "peak_basics",
        "peak_positions",
        "peak_proximity",
        "peak_shadow",
        "peak_ambience",
        "peak_se_density",
        "peak_nearest_triggering",
    )
    provides = "isolated_s1"
    data_kind = "isolated_s1"

    isolated_peaks_fields = straxen.URLConfig(
        default=list(np.dtype(positioned_peak_dtype()).names) + correlation_fields,
        type=(list, tuple),
        help="Needed fields in isolated peaks",
    )

    groups_seen = 0

    def refer_dtype(self, data_kind):
        provided_fields, union_dtype = needed_dtype(
            self.deps, [self.dependencies_by_kind()[data_kind]], set.union
        )
        return strax.unpack_dtype(union_dtype)

    def infer_dtype(self):
        dtype_reference = self.refer_dtype("peaks")
        dtype = copy_dtype(dtype_reference, self.isolated_peaks_fields)
        dtype += [
            (("Run id", "run_id"), np.int32),
            (("Group number of peaks", "group_number"), np.int64),
        ]
        return dtype

    def compute(self, peaks):
        _result = peaks[peaks["cut_isolated_s1"]]
        result = np.empty(len(_result), dtype=self.dtype)
        for n in result.dtype.names:
            if n not in ["run_id", "group_number"]:
                result[n] = _result[n]

        result["run_id"] = int(self.run_id)
        result["group_number"] = np.arange(len(result)) + self.groups_seen

        self.groups_seen += len(result)
        return result
