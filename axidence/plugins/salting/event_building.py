import numpy as np
import strax
from strax import Plugin
from straxen import Events


class SaltedEvents(Events):
    __version__ = "0.0.0"
    depends_on = ("salting_peaks", "salting_peak_proximity", "peak_basics", "peak_proximity")
    provides = "events"
    data_kind = "events"
    save_when = strax.SaveWhen.EXPLICIT

    dtype = [
        ("event_number", np.int64, "Event number in this dataset"),
        ("time", np.int64, "Event start time in ns since the unix epoch"),
        ("endtime", np.int64, "Event end time in ns since the unix epoch"),
    ]

    def compute(self, salting_peaks, peaks, start, end):
        pass


class SaltedEventBasics(Plugin):
    __version__ = "0.0.0"
    depends_on = ("events", "salting_peaks", "peak_basics", "peak_positions")
    provides = "event_basics"
    data_kind = "events"
    save_when = strax.SaveWhen.EXPLICIT

    dtype = strax.time_fields
