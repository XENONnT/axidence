import numpy as np
import strax
from straxen import (
    PeakProximity,
    PeakShadow,
    PeakAmbience,
)

from ...utils import copy_dtype


class PeakProximitySalted(PeakProximity):
    __version__ = "0.0.0"
    child_plugin = True
    depends_on = ("peaks_salted", "peak_basics", "peak_positions")
    provides = "peak_proximity_salted"
    data_kind = "peaks_salted"
    save_when = strax.SaveWhen.EXPLICIT

    def refer_dtype(self):
        return strax.unpack_dtype(strax.to_numpy_dtype(super(PeakProximitySalted, self).dtype))

    def infer_dtype(self):
        dtype_reference = self.refer_dtype()
        available = {d[0][1] for d in dtype_reference}
        # `proximity_score` was added to straxen PeakProximity after SR1.
        candidate = ["time", "endtime", "proximity_score", "n_competing_left", "n_competing"]
        required_names = [n for n in candidate if n in available]
        dtype = copy_dtype(dtype_reference, required_names)
        # since event_number is int64 in event_basics
        dtype += [
            (("Salting number of peaks", "salt_number"), np.int64),
        ]
        return dtype

    def compute(self, peaks_salted, peaks):
        # SR1 PeakProximity has no `compute_proximity(peaks, current_peak)` helper.
        # Build a sorted union of (real peaks + salted peaks), run the SR1 compute
        # on it, then pick out the salted rows in their original order.
        n_salted = len(peaks_salted)
        if n_salted == 0:
            return np.zeros(0, dtype=self.dtype)

        merged_dtype = [
            ("time", np.int64),
            ("endtime", np.int64),
            ("area", np.float32),
        ]
        merged = np.empty(len(peaks) + n_salted, dtype=merged_dtype)
        merged["time"][: len(peaks)] = peaks["time"]
        merged["endtime"][: len(peaks)] = peaks["endtime"]
        merged["area"][: len(peaks)] = peaks["area"]
        merged["time"][len(peaks) :] = peaks_salted["time"]
        merged["endtime"][len(peaks) :] = peaks_salted["endtime"]
        merged["area"][len(peaks) :] = peaks_salted["area"]
        order = np.argsort(merged["time"])
        merged = merged[order]
        # invert the permutation so we can recover salted-row positions in the sort
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        salted_in_sorted = inv[len(peaks) :]

        proximity_dict = super().compute(merged)

        result = np.zeros(n_salted, dtype=self.dtype)
        result["time"] = peaks_salted["time"]
        result["endtime"] = peaks_salted["endtime"]
        for name in result.dtype.names:
            if name in ("time", "endtime", "salt_number"):
                continue
            if name in proximity_dict:
                result[name] = np.asarray(proximity_dict[name])[salted_in_sorted]
        result["salt_number"] = peaks_salted["salt_number"]
        # here the plus one accounts for the peak itself
        if "n_competing" in result.dtype.names:
            result["n_competing"] += 1
        return result


class PeakShadowSalted(PeakShadow):
    __version__ = "0.0.0"
    child_plugin = True
    depends_on = ("peaks_salted", "peak_basics", "peak_positions")
    provides = "peak_shadow_salted"
    data_kind = "peaks_salted"
    save_when = strax.SaveWhen.EXPLICIT

    def infer_dtype(self):
        dtype = super().infer_dtype()
        dtype += [
            (("Salting number of peaks", "salt_number"), np.int64),
        ]
        return dtype

    def compute(self, peaks_salted, peaks):
        result = self.compute_shadow(peaks, peaks_salted)
        result["salt_number"] = peaks_salted["salt_number"]
        return result


class PeakAmbienceSalted(PeakAmbience):
    __version__ = "0.0.0"
    child_plugin = True
    depends_on = ("peaks_salted", "lone_hits", "peak_basics", "peak_positions")
    provides = "peak_ambience_salted"
    data_kind = "peaks_salted"
    save_when = strax.SaveWhen.EXPLICIT

    def infer_dtype(self):
        dtype = super().infer_dtype()
        dtype += [
            (("Salting number of peaks", "salt_number"), np.int64),
        ]
        return dtype

    def compute(self, peaks_salted, lone_hits, peaks):
        result = self.compute_ambience(lone_hits, peaks, peaks_salted)
        result["salt_number"] = peaks_salted["salt_number"]
        return result


# SR0 release: PeakNearestTriggeringSalted and PeakSEScoreSalted are
# dropped because the underlying straxen.PeakNearestTriggering /
# PeakSEScore plugins don't exist in straxen 1.7.x.
