import numpy as np
import strax
from strax import OverlapWindowPlugin
import straxen

from ...utils import copy_dtype

export, __all__ = strax.exporter()


class EventsForcePaired(OverlapWindowPlugin):
    """Fake event, mimicking Events Force manually pairing of isolated S1 & S2
    Actually NOT used in AC simulation."""

    depends_on = "peaks_paired"
    provides = "events_paired"
    data_kind = "events_paired"
    save_when = strax.SaveWhen.EXPLICIT

    paring_event_interval = straxen.URLConfig(
        default=int(1e8),
        type=int,
        help="The interval which separates two events S1 [ns]",
    )

    def infer_dtype(self):
        dtype_reference = strax.unpack_dtype(self.deps["peaks_paired"].dtype_for("peaks_paired"))
        required_names = ["time", "endtime", "event_number"]
        dtype = copy_dtype(dtype_reference, required_names)
        return dtype

    def get_window_size(self):
        return 10 * self.paring_event_interval

    def compute(self, peaks_paired):
        peaks_event_number_sorted = np.sort(peaks_paired, order=("event_number", "time"))
        event_number, event_number_index, event_number_count = np.unique(
            peaks_event_number_sorted["event_number"], return_index=True, return_counts=True
        )
        event_number_index = np.append(event_number_index, len(peaks_event_number_sorted))
        result = np.zeros(len(event_number), self.dtype)
        result["time"] = peaks_event_number_sorted["time"][event_number_index[:-1]]
        result["endtime"] = strax.endtime(peaks_event_number_sorted)[event_number_index[1:] - 1]
        result["event_number"] = event_number
        return result
