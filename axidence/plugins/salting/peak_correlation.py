import numba
import numpy as np
import strax
from straxen import (
    PeakProximity,
    PeakShadow,
    PeakAmbience,
    PeakNearestTriggering,
    PeakSEScore,
)

from ...utils import copy_dtype


class PeakProximitySalted(PeakProximity):
    __version__ = "0.0.1"
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
        # SR1 PeakProximity has no `compute_proximity(peaks, current_peak)` helper,
        # so count the competing *real* peaks around each salted peak ourselves.
        # Two conventions matter here and both follow SR1 straxen 2.2.7:
        #   * `PeakProximity.find_n_competing` does NOT count the peak itself
        #     (modern straxen does, which is why `main` adds one), so no +1 here;
        #   * only real peaks compete, the salted partner peak (S1 of a salted S2
        #     or vice versa) is not counted, as in axidence v0.3.x.
        # `Events._is_triggering` in SR1 cuts on `n_competing <= trigger_max_competing`,
        # so any offset here directly biases which salted S2s can trigger.
        if "proximity_score" in self.dtype.names:
            raise NotImplementedError(
                "proximity_score is not available on the SR1 (straxen 2.2.x) stack."
            )
        windows = strax.touching_windows(peaks, peaks_salted, window=self.nearby_window)
        n_left, n_tot = self.find_n_competing_salted(
            peaks, peaks_salted, windows, fraction=self.min_area_fraction
        )
        return dict(
            time=peaks_salted["time"],
            endtime=strax.endtime(peaks_salted),
            n_competing_left=n_left,
            n_competing=n_tot,
            salt_number=peaks_salted["salt_number"],
        )

    @staticmethod
    @numba.jit(nopython=True, nogil=True, cache=True)
    def find_n_competing_salted(peaks, peaks_salted, windows, fraction):
        """Number of real peaks larger than `fraction` of each salted peak's area within its
        touching window, split into (left of the salted peak, total)."""
        n_left = np.zeros(len(peaks_salted), dtype=np.int32)
        n_tot = n_left.copy()
        areas = peaks["area"]
        areas_salted = peaks_salted["area"]

        dig = np.searchsorted(peaks["center_time"], peaks_salted["center_time"])

        for i, peak in enumerate(peaks_salted):
            left_i, right_i = windows[i]
            threshold = areas_salted[i] * fraction
            n_left[i] = np.sum(areas[left_i : dig[i]] > threshold)
            n_tot[i] = n_left[i] + np.sum(areas[dig[i] : right_i] > threshold)

        return n_left, n_tot


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


class PeakNearestTriggeringSalted(PeakNearestTriggering):
    __version__ = "0.0.0"
    child_plugin = True
    depends_on = (
        "peaks_salted",
        "peak_proximity_salted",
        "peak_basics",
        "peak_proximity",
    )
    provides = "peak_nearest_triggering_salted"
    data_kind = "peaks_salted"
    save_when = strax.SaveWhen.EXPLICIT

    def infer_dtype(self):
        dtype = super().infer_dtype()
        dtype += [
            (("Salting number of peaks", "salt_number"), np.int64),
        ]
        return dtype

    def compute(self, peaks_salted, peaks):
        result = self.compute_triggering(peaks, peaks_salted)
        result["salt_number"] = peaks_salted["salt_number"]
        return result


class PeakSEScoreSalted(PeakSEScore):
    __version__ = "0.0.0"
    child_plugin = True
    depends_on = ("peaks_salted", "peak_basics", "peak_positions")
    provides = "peak_se_score_salted"
    data_kind = "peaks_salted"
    save_when = strax.SaveWhen.EXPLICIT

    def infer_dtype(self):
        dtype = super().infer_dtype()
        dtype += [
            (("Salting number of peaks", "salt_number"), np.int64),
        ]
        return dtype

    def compute(self, peaks_salted, peaks):
        se_score = self.compute_se_score(peaks, peaks_salted)
        return dict(
            time=peaks_salted["time"], endtime=strax.endtime(peaks_salted), se_score=se_score
        )
