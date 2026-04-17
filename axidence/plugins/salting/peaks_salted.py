import numpy as np
import strax
import straxen
from straxen import PeakBasics

from ...utils import copy_dtype


class PeaksSalted(PeakBasics):
    __version__ = "0.0.1"
    child_plugin = True
    depends_on = "events_salting"
    provides = "peaks_salted"
    data_kind = "peaks_salted"
    save_when = strax.SaveWhen.EXPLICIT

    only_salt_s1 = straxen.URLConfig(
        default=False,
        type=bool,
        help="Whether only salt S1",
    )

    only_salt_s2 = straxen.URLConfig(
        default=False,
        type=bool,
        help="Whether only salt S2",
    )

    def refer_dtype(self):
        return strax.unpack_dtype(strax.to_numpy_dtype(super(PeaksSalted, self).infer_dtype()))

    def infer_dtype(self):
        dtype_reference = self.refer_dtype()
        required_names = ["time", "endtime", "center_time"]
        required_names += ["area", "area_fraction_top", "n_hits", "tight_coincidence", "type"]
        dtype = copy_dtype(dtype_reference, required_names)
        # since event_number is int64 in event_basics
        dtype += [
            (("Reconstructed S2 X position (cm), uncorrected", "x"), np.float32),
            (("Reconstructed S2 Y position (cm), uncorrected", "y"), np.float32),
            (("Salting number of peaks", "salt_number"), np.int64),
        ]
        return dtype

    def setup(self):
        super().setup()
        if self.only_salt_s1 and self.only_salt_s2:
            raise ValueError("Cannot only salt both S1 and S2.")

    def compute(self, events_salting):
        """Copy features of events_salting into peaks_salted."""
        peaks_salted = np.empty(len(events_salting) * 2, dtype=self.dtype)
        for n in "center_time area".split():
            peaks_salted[n] = np.vstack(
                [
                    events_salting[f"s1_{n}"],
                    events_salting[f"s2_{n}"],
                ]
            ).T.flatten()
        peaks_salted["time"] = peaks_salted["center_time"]
        # add one to prevent error about non-positive length
        peaks_salted["endtime"] = peaks_salted["time"] + 1
        for n in "n_hits tight_coincidence".split():
            peaks_salted[n] = np.vstack(
                [
                    events_salting[f"s1_{n}"],
                    np.full(len(events_salting), -1),
                ]
            ).T.flatten()
        for n in "x y area_fraction_top".split():
            peaks_salted[n] = np.vstack(
                [
                    np.full(len(events_salting), np.nan),
                    events_salting[f"s2_{n}"],
                ]
            ).T.flatten()
        peaks_salted["type"] = np.vstack(
            [
                np.full(len(events_salting), 1),
                np.full(len(events_salting), 2),
            ]
        ).T.flatten()
        peaks_salted["salt_number"] = np.repeat(events_salting["salt_number"], 2)

        # Filter out peaks that are not S1 or S2
        if self.only_salt_s1:
            peaks_salted = peaks_salted[peaks_salted["type"] == 1]
        if self.only_salt_s2:
            peaks_salted = peaks_salted[peaks_salted["type"] == 2]
        return peaks_salted
