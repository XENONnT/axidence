from typing import Tuple
import numpy as np
import strax
from strax import Plugin
import straxen


class SaltingPeaks(Plugin):
    __version__ = "0.0.0"
    depends_on: Tuple = tuple()
    provides = "salting_peaks"
    data_kind = "salting_peaks"
    save_when = strax.SaveWhen.EXPLICIT

    dtype = [
        (("Start time of the peak (ns since unix epoch)", "time"), np.int64),
        (("End time of the peak (ns since unix epoch)", "endtime"), np.int64),
        (("Weighted center time of the peak (ns since unix epoch)", "center_time"), np.int64),
        (("Peak integral in PE", "area"), np.float32),
        (("Number of hits contributing at least one sample to the peak", "n_hits"), np.int32),
        (("Number of PMTs contributing to the peak", "n_channels"), np.int16),
        (("Length of the peak waveform in samples", "length"), np.int32),
        (("Time resolution of the peak waveform in ns", "dt"), np.int16),
        (
            ("Number of PMTs with hits within tight range of mean", "tight_coincidence"),
            np.int16,
        ),
        (("Classification of the peak(let)", "type"), np.int8),
        (("Salting number of events", "salt_number"), np.int16),
    ]

    source_done = False

    dtype = strax.time_fields

    salting_rate = straxen.URLConfig(
        default=10,
        type=(int, float),
        help="Rate of salting in Hz",
    )

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

    s1_distribution = straxen.URLConfig(
        default="",
        type=str,
        help="S1 distribution shape",
    )

    s2_distribution = straxen.URLConfig(
        default="",
        type=str,
        help="S2 distribution shape",
    )

    def setup(self):
        pass

    def compute(self):
        self.source_done = True
        # return self.chunk(start=start, end=end, data=result)

    def source_finished(self):
        return self.source_done

    def is_ready(self, chunk_i):
        if "ready" not in self.__dict__:
            self.ready = False
        self.ready ^= True  # Flip
        return self.ready
