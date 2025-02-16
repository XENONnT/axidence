import numpy as np
import strax
from straxen import PeakProximity, PeakShadow, PeakAmbience, PeakNearestTriggering, PeakSEScore

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
        mask = np.isin(peaks["type"], [1, 2])
        windows = strax.touching_windows(peaks[mask], peaks_salted, window=self.nearby_window)
        n_left, n_tot = self.find_n_competing(
            peaks[mask], peaks_salted, windows, fraction=self.min_area_fraction
        )
        return dict(
            time=peaks_salted["time"],
            endtime=strax.endtime(peaks_salted),
            # here the plus one accounts for the peak itself
            n_competing=n_tot + 1,
            n_competing_left=n_left,
            salt_number=peaks_salted["salt_number"],
        )


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
