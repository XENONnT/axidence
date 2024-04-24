import numpy as np
from tqdm import tqdm
import strax
from strax import Plugin
import straxen
from straxen import Events, EventBasics

from axidence.plugin import ExhaustPlugin


class SaltedEvents(Events, ExhaustPlugin):
    __version__ = "0.0.0"
    depends_on = ("salting_peaks", "salting_peak_proximity", "peak_basics", "peak_proximity")
    provides = "events"
    data_kind = "events"
    save_when = strax.SaveWhen.EXPLICIT

    n_drift_time_window = straxen.URLConfig(
        default=5,
        type=int,
        track=True,
        help="How many max drift time will the event builder extend",
    )

    disable_tqdm = straxen.URLConfig(
        default=True,
        type=bool,
        track=False,
        help="Whether to disable tqdm",
    )

    def __init__(self):
        super().__init__()
        self.dtype = super().dtype + [
            (("Salting number of events", "salt_number"), np.int64),
            (("Whether the salting event can trigger", "is_triggering"), bool),
        ]

    def setup(self):
        super().setup()

        # intersection depends_on's dtype.names will be needed in event building
        self.needed_fields = set.intersection(
            *tuple(
                set.union(*tuple(set(self.deps[d].dtype_for(d).names) for d in dk))
                for dk in self.dependencies_by_kind().values()
            )
        )
        dk = list(self.dependencies_by_kind().values())[0]
        dtype_reference = strax.merged_dtype([self.deps[d].dtype_for(d) for d in dk])
        self._peaks_dtype = []
        for x in dtype_reference:
            if x[0][1] in self.needed_fields:
                self._peaks_dtype.append(x)
        if len(self._peaks_dtype) != len(self.needed_fields):
            raise ValueError(
                "Weird! Could not find all needed fields "
                f"{self.needed_fields} in {dtype_reference}!"
            )

        self.window = self.n_drift_time_window * self.drift_time_max

        if self.window < self.left_extension + self.right_extension:
            raise ValueError(
                f"The window {self.window}ns is too small to extend the event "
                f"while the gap_threshold is about {self.left_extension + self.right_extension}!"
            )

    def get_window_size(self):
        return max(super().get_window_size(), self.window * 10)

    def compute(self, salting_peaks, peaks, start, end):
        if salting_peaks["salt_number"][0] != 0:
            raise ValueError(
                "Expected salt_number to start from 0 because "
                f"{self.__class__.__name__} is a ExhaustPlugin plugin!"
            )

        n_events = len(salting_peaks) // 2
        if np.unique(salting_peaks["salt_number"]).size != n_events:
            raise ValueError("Expected salt_number to be half of the input peaks number!")

        # combine salting_peaks and peaks
        _peaks = np.empty(len(salting_peaks) + len(peaks), dtype=self._peaks_dtype)
        for n in set(self.needed_fields):
            _peaks[n] = np.hstack([salting_peaks[n], peaks[n]])
        _peaks = np.sort(_peaks, order="time")
        # use S2s as anchors
        anchor_peaks = salting_peaks[1::2]
        windows = strax.touching_windows(_peaks, anchor_peaks, window=self.window)

        # check if the salting event can trigger
        # for now, only S2 can trigger
        if not self.exclude_s1_as_triggering_peaks:
            raise NotImplementedError("Only S2 can trigger for now!")
        result = np.empty(n_events, self.dtype)
        if np.unique(anchor_peaks["type"]).size != 1:
            raise ValueError("Expected only one type of anchor peaks!")

        # prepare for an empty event
        empty_event = np.empty(1, dtype=self.dtype)
        EventBasics.set_nan_defaults(empty_event)

        # iterate through all anchor peaks
        _result = []
        for i in tqdm(range(n_events), disable=self.disable_tqdm):
            left_i, right_i = windows[i]
            _events = super().compute(_peaks[left_i:right_i], start, end)
            _events = strax.split_touching_windows(_events, anchor_peaks[i : i + 1])
            if result["is_triggering"][i]:
                if len(_events) != 1 or len(_events[0]) != 1:
                    raise ValueError(f"Expected 1 event, got {_events}!")
                _result.append(_events[0])
            else:
                empty_event["time"] = anchor_peaks["time"][i]
                empty_event["endtime"] = anchor_peaks["endtime"][i]
                _result.append(empty_event)
        _result = np.hstack(_result)

        for n in _result.dtype.names:
            result[n] = _result[n]

        # assign the most important parameters
        result["is_triggering"] = self._is_triggering(anchor_peaks)
        result["salt_number"] = salting_peaks["salt_number"][::2]
        result["event_number"] = salting_peaks["salt_number"][::2]
        return result


class SaltedEventBasics(Plugin):
    __version__ = "0.0.0"
    depends_on = ("events", "salting_peaks", "peak_basics", "peak_positions")
    provides = "event_basics"
    data_kind = "events"
    save_when = strax.SaveWhen.EXPLICIT

    dtype = strax.time_fields