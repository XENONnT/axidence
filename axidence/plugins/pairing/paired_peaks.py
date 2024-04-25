from immutabledict import immutabledict
import numpy as np
import strax
import straxen

from ...plugin import ExhaustPlugin


class PairedPeaks(ExhaustPlugin):
    __version__ = "0.0.0"
    depends_on = ("isolated_s1", "isolated_s2")
    provides = ("paired_peaks", "paired_truth")
    data_kind = immutabledict(zip(provides, provides))

    isolated_peaks_fields = straxen.URLConfig(
        default=np.dtype(strax.peak_dtype(n_channels=straxen.n_tpc_pmts)).names,
        type=(list, tuple),
        help="Needed fields in isolated peaks",
    )

    isolated_events_fields = straxen.URLConfig(
        default=[],
        type=(list, tuple),
        help="Needed fields in isolated events",
    )

    def infer_dtype(self):
        dtype = self.deps["isolated_s2"].dtype_for("isolated_s2")
        peaks_dtype = dtype + [
            (("Event number in this dataset", "event_number"), np.int32),
            # (("Original run id", "origin_run_id"), np.int32),
            (("Original isolated S1/S2 group", "origin_group_number"), np.int32),
            (("Original time of peaks", "origin_time"), np.int64),
            (("Original endtime of peaks", "origin_endtime"), np.int64),
            (("Original center_time of peaks", "origin_center_time"), np.int64),
            (("Original n_competing", "origin_n_competing"), np.int32),
            (("Original type of group", "origin_group_type"), np.int8),
            (("Original s1_index in isolated S2", "origin_s1_index"), np.int32),
            (("Original s2_index in isolated S2", "origin_s2_index"), np.int32),
        ]
        truth_dtype = [
            (("Event number in this dataset", "event_number"), np.int32),
            (("Drift time between InpS1 and main InpS2 in ns", "drift_time"), np.float32),
            # (("Original run id of isolated S1", "s1_run_id"), np.int32),
            # (("Original run id of isolated S2", "s2_run_id"), np.int32),
            (("Original isolated S1 group", "s1_group_number"), np.int32),
            (("Original isolated S2 group", "s2_group_number"), np.int32),
        ] + strax.time_fields
        return dict(paired_peaks=peaks_dtype, paired_truth=truth_dtype)
