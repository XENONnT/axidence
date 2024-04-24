import numpy as np
import strax
import straxen
from straxen import PeakBasics

from ...utils import copy_dtype


class SaltingPeaks(PeakBasics):
    __version__ = "0.0.0"
    depends_on = "salting_events"
    provides = "salting_peaks"
    data_kind = "salting_peaks"
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
        # merged_dtype is needed because the refer_dtype should return a list
        return strax.merged_dtype(
            [
                strax.to_numpy_dtype(super(SaltingPeaks, self).infer_dtype()),
            ]
        )

    def infer_dtype(self):
        dtype_reference = self.refer_dtype()
        required_names = ["time", "endtime", "center_time"]
        required_names += ["area", "n_hits", "tight_coincidence", "type"]
        dtype = copy_dtype(dtype_reference, required_names)
        # since event_number is int64 in event_basics
        dtype += [
            ("x", np.float32, "Reconstructed S2 X position (cm), uncorrected"),
            ("y", np.float32, "Reconstructed S2 Y position (cm), uncorrected"),
            (("Salting number of peaks", "salt_number"), np.int64),
        ]
        return dtype

    def compute(self, salting_events):
        """Copy features of salting_events into salting_peaks."""
        salting_peaks = np.empty(len(salting_events) * 2, dtype=self.dtype)
        for n in "center_time area".split():
            salting_peaks[n] = np.vstack(
                [
                    salting_events[f"s1_{n}"],
                    salting_events[f"s2_{n}"],
                ]
            ).T.flatten()
        salting_peaks["time"] = salting_peaks["center_time"]
        # add one to prevent error about non-positive length
        salting_peaks["endtime"] = salting_peaks["time"] + 1
        for n in "n_hits tight_coincidence".split():
            salting_peaks[n] = np.vstack(
                [
                    salting_events[f"s1_{n}"],
                    np.full(len(salting_events), -1),
                ]
            ).T.flatten()
        for n in "x y".split():
            salting_peaks[n] = np.vstack(
                [
                    np.full(len(salting_events), np.nan),
                    salting_events[f"s2_{n}"],
                ]
            ).T.flatten()
        salting_peaks["type"] = np.vstack(
            [
                np.full(len(salting_events), 1),
                np.full(len(salting_events), 2),
            ]
        ).T.flatten()
        salting_peaks["salt_number"] = np.repeat(salting_events["salt_number"], 2)
        return salting_peaks
