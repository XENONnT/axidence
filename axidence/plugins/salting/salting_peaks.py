import numpy as np
import strax
from strax import Plugin
import straxen
from straxen.misc import kind_colors


kind_colors["salting_peaks"] = "#00ffff"


class SaltingPeaks(Plugin):
    __version__ = "0.0.0"
    depends_on = "salting_events"
    provides = "salting_peaks"
    data_kind = "salting_peaks"
    save_when = strax.SaveWhen.EXPLICIT

    dtype = [
        (("Start time of the peak (ns since unix epoch)", "time"), np.int64),
        (("End time of the peak (ns since unix epoch)", "endtime"), np.int64),
        (("Weighted center time of the peak (ns since unix epoch)", "center_time"), np.int64),
        (("Peak integral in PE", "area"), np.float32),
        (("Number of hits contributing at least one sample to the peak", "n_hits"), np.int32),
        (
            ("Number of PMTs with hits within tight range of mean", "tight_coincidence"),
            np.int16,
        ),
        (("Classification of the peak(let)", "type"), np.int8),
        (("Salting number of events", "salt_number"), np.int16),
    ]

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
