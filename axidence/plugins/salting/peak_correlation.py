import numba
import numpy as np
import strax
from straxen import PeakProximity, PeakShadow, PeakAmbience, PeakNearestTriggering, PeakSEDensity

from ...utils import copy_dtype


class PeakProximitySalted(PeakProximity):
    __version__ = "0.0.0"
    child_plugin = True
    depends_on = ("peaks_salted", "peak_basics")
    provides = "peak_proximity_salted"
    data_kind = "peaks_salted"
    save_when = strax.SaveWhen.EXPLICIT

    def refer_dtype(self):
        return strax.unpack_dtype(strax.to_numpy_dtype(super(PeakProximitySalted, self).dtype))

    def infer_dtype(self):
        dtype_reference = self.refer_dtype()
        required_names = ["time", "endtime", "n_competing", "n_competing_left"]
        dtype = copy_dtype(dtype_reference, required_names)
        # since event_number is int64 in event_basics
        dtype += [
            (("Salting number of peaks", "salt_number"), np.int64),
        ]
        return dtype

    def compute(self, peaks_salted, peaks):
        windows = strax.touching_windows(peaks, peaks_salted, window=self.nearby_window)
        n_left, n_tot = self.find_n_competing(
            peaks, peaks_salted, windows, fraction=self.min_area_fraction
        )
        return dict(
            time=peaks_salted["time"],
            endtime=strax.endtime(peaks_salted),
            n_competing=n_tot,
            n_competing_left=n_left,
            salt_number=peaks_salted["salt_number"],
        )

    @staticmethod
    @numba.jit(nopython=True, nogil=True, cache=True)
    def find_n_competing(peaks, peaks_salted, windows, fraction):
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
    depends_on = ("peaks_salted", "peak_proximity_salted", "peak_basics", "peak_proximity")
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


class PeakSEDensitySalted(PeakSEDensity):
    __version__ = "0.0.0"
    child_plugin = True
    depends_on = ("peaks_salted", "peak_basics", "peak_positions")
    provides = "peak_se_density_salted"
    data_kind = "peaks_salted"
    save_when = strax.SaveWhen.EXPLICIT

    def infer_dtype(self):
        dtype = super().infer_dtype()
        dtype += [
            (("Salting number of peaks", "salt_number"), np.int64),
        ]
        return dtype

    def compute(self, peaks_salted, peaks):
        se_density = self.compute_se_density(peaks, peaks_salted)
        return dict(
            time=peaks_salted["time"], endtime=strax.endtime(peaks_salted), se_density=se_density
        )
