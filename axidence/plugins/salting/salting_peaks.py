import numpy as np
import strax
import straxen
from straxen import PeakBasics, PeakPositionsNT


class SaltingPeaks(PeakBasics, PeakPositionsNT):
    __version__ = "0.0.0"
    depends_on = "salting_events"
    provides = "salting_peaks"
    data_kind = "salting_peaks"
    save_when = strax.SaveWhen.EXPLICIT

    def infer_dtype(self):
        dtype = []
        dtype_reference = strax.merged_dtype(
            [
                strax.to_numpy_dtype(super().infer_dtype()),
                strax.to_numpy_dtype(super(PeakPositionsNT, self).infer_dtype()),
            ]
        )
        for n in (
            "time",
            "endtime",
            "center_time",
            "area",
            "n_hits",
            "tight_coincidence",
            "x",
            "y",
            "type",
        ):
            for x in dtype_reference:
                found = False
                if (x[0][1] == n) and (not found):
                    dtype.append(x)
                    found = True
                    break
            if not found:
                raise ValueError(f"Could not find {n} in dtype_reference!")
        # since event_number is int64 in event_basics
        dtype += [(("Salting number of peaks", "salt_number"), np.int64)]
        return dtype

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

    def compute(self, salting_events):
        salting_peaks = np.empty(len(salting_events) * 2, dtype=self.dtype)
        for n in "time endtime".split():
            salting_peaks[n] = np.repeat(salting_events[n], 2)
        for n in "center_time area".split():
            salting_peaks[n] = np.vstack(
                [
                    salting_events[f"s1_{n}"],
                    salting_events[f"s2_{n}"],
                ]
            ).T.flatten()
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
