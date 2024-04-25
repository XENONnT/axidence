from typing import Tuple
import numpy as np
import strax
import straxen
from straxen import Events, EventBasics

from ...utils import needed_dtype, merge_salting_real
from axidence.plugin import ExhaustPlugin


class SaltedEvents(Events, ExhaustPlugin):
    __version__ = "0.1.0"
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

    def __init__(self):
        super().__init__()
        self.dtype = super().dtype + [
            (("Salting number of events", "salt_number"), np.int64),
            (("Whether the salting event can trigger", "is_triggering"), bool),
        ]

    def setup(self):
        super().setup()
        self.needed_fields, self._peaks_dtype = needed_dtype(
            self.deps, self.dependencies_by_kind, set.intersection
        )

        self.window = self.n_drift_time_window * self.drift_time_max

        if self.window < self.left_extension + self.right_extension:
            raise ValueError(
                f"The window {self.window}ns is too small to extend the event "
                f"while the gap_threshold is about {self.left_extension + self.right_extension}!"
            )

        # for now, only S2 can trigger
        if not self.exclude_s1_as_triggering_peaks:
            raise NotImplementedError("Only S2 can trigger for now!")

    def get_window_size(self):
        return max(super().get_window_size(), self.window * 10)

    def compute(self, salting_peaks, peaks, start, end):
        if salting_peaks["salt_number"][0] != 0:
            raise ValueError(
                "Expected salt_number to start from 0 because "
                f"{self.__class__.__name__} is a ExhaustPlugin plugin!"
            )

        _peaks = merge_salting_real(salting_peaks, peaks, self._peaks_dtype)

        # use S2s as anchors
        anchor_peaks = salting_peaks[1::2]
        if np.unique(anchor_peaks["type"]).size != 1:
            raise ValueError("Expected only one type of anchor peaks!")

        # initial the final result
        n_events = len(salting_peaks) // 2
        if np.unique(salting_peaks["salt_number"]).size != n_events:
            raise ValueError("Expected salt_number to be half of the input peaks number!")
        result = np.empty(n_events, self.dtype)

        # check if the salting anchor can trigger
        is_triggering = self._is_triggering(anchor_peaks)

        # prepare for an empty event
        empty_events = np.empty(len(anchor_peaks), dtype=self.dtype)
        empty_events["time"] = anchor_peaks["time"]
        empty_events["endtime"] = anchor_peaks["endtime"]

        # build events at once because if a messy environment
        # make the two anchor in the same event
        # it will be considered as event building failure later
        events = super().compute(_peaks, start, end)
        events = strax.split_touching_windows(events, anchor_peaks)

        # there should be only one event near the anchor
        l_events = np.array([len(event) for event in events])
        if not np.all(l_events <= 1):
            raise ValueError(f"Expected one event per anchor, got {l_events.max()}!")

        # merge events and placeholders
        _result = np.hstack(
            [events[i] if l_events[i] != 0 else empty_events[i] for i in range(n_events)]
        )
        # this more fancy way will not be used
        # _result = np.sort(np.hstack(events + [empty_events[l_events == 0]]), order="time")
        if not np.all(np.diff(_result["time"]) >= 0):
            raise ValueError("Expected the result to be sorted!")
        for n in _result.dtype.names:
            result[n] = _result[n]

        # assign the most important parameters
        result["is_triggering"] = is_triggering
        result["salt_number"] = salting_peaks["salt_number"][::2]
        result["event_number"] = salting_peaks["salt_number"][::2]

        if np.any(np.diff(result["time"]) < 0):
            raise ValueError("Expected time to be sorted!")
        return result


class SaltedEventBasics(EventBasics, ExhaustPlugin):
    __version__ = "0.0.0"
    depends_on: Tuple[str, ...] = (
        "events",
        "salting_peaks",
        "salting_peak_proximity",
        "peak_basics",
        "peak_proximity",
        "peak_positions",
    )
    provides = "event_basics"
    data_kind = "events"
    save_when = strax.SaveWhen.EXPLICIT

    def infer_dtype(self):
        dtype = super().infer_dtype()
        dtype += [
            (("Salting number of main S1", "s1_salt_number"), np.int64),
            (("Salting number of main S2", "s2_salt_number"), np.int64),
            (("Salting number of alternative S1", "alt_s1_salt_number"), np.int64),
            (("Salting number of alternative S2", "alt_s2_salt_number"), np.int64),
            (("Salting number of events", "salt_number"), np.int64),
            (("Whether the salting event can trigger", "is_triggering"), bool),
        ]
        return dtype

    def setup(self):
        super().setup()

        self.needed_fields, self._peaks_dtype = needed_dtype(
            self.deps, self.dependencies_by_kind, set.union
        )

        self.peak_properties = tuple(
            list(self.peak_properties) + [("salt_number", np.int64, "Salting number of peaks")]
        )

    def compute(self, events, salting_peaks, peaks):
        if salting_peaks["salt_number"][0] != 0:
            raise ValueError(
                "Expected salt_number to start from 0 because "
                f"{self.__class__.__name__} is a ExhaustPlugin plugin!"
            )

        _peaks = merge_salting_real(salting_peaks, peaks, self._peaks_dtype)

        result = np.zeros(len(events), dtype=self.dtype)
        self.set_nan_defaults(result)

        split_peaks = strax.split_by_containment(_peaks, events)

        result["time"] = events["time"]
        result["endtime"] = events["endtime"]
        result["salt_number"] = events["salt_number"]
        result["event_number"] = events["event_number"]

        self.fill_events(result, events, split_peaks)
        result["is_triggering"] = events["is_triggering"]

        if np.all(result["s1_salt_number"] < 0) or np.all(result["s2_salt_number"] < 0):
            raise ValueError("Found zero triggered salting peaks!")
        return result
