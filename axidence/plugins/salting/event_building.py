import warnings
from typing import Tuple
import numpy as np
import strax
from strax import ExhaustPlugin
import straxen
from straxen import Events, EventBasics

from ...utils import needed_dtype, merge_salted_real


class EventsSalted(Events, ExhaustPlugin):
    __version__ = "0.2.0"
    child_plugin = True
    depends_on = ("peaks_salted", "peak_proximity_salted", "peak_basics", "peak_proximity")
    provides = "events_salted"
    data_kind = "events_salted"
    save_when = strax.SaveWhen.EXPLICIT

    n_drift_time_window = straxen.URLConfig(
        default=5,
        type=int,
        help="How many max drift time will the event builder extend",
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

    def __init__(self):
        super().__init__()
        self.dtype = super().dtype + [
            (("Salting number of events", "salt_number"), np.int64),
            (("Whether the salting event can trigger", "is_triggering"), bool),
        ]

    def setup(self):
        super().setup()
        self.needed_fields, self._peaks_dtype = needed_dtype(
            self.deps, self.dependencies_by_kind().values(), set.intersection
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

    def compute(self, peaks_salted, peaks, start, end):
        if peaks_salted["salt_number"][0] != 0:
            raise ValueError(
                "Expected salt_number to start from 0 because "
                f"{self.__class__.__name__} is a ExhaustPlugin plugin!"
            )

        _peaks = merge_salted_real(peaks_salted, peaks, self._peaks_dtype)

        if self.only_salt_s1 or self.only_salt_s2:
            anchor_peaks = peaks_salted
        else:
            # use S2s as anchors by default
            anchor_peaks = peaks_salted[1::2]

        if np.unique(anchor_peaks["type"]).size != 1:
            raise ValueError("Expected only one type of anchor peaks!")

        # initial the final result
        if self.only_salt_s1 or self.only_salt_s2:
            n_events = len(peaks_salted)
        else:
            n_events = len(peaks_salted) // 2
        if np.unique(peaks_salted["salt_number"]).size != n_events:
            raise ValueError("Expected salt_number to be half of the input peaks number!")
        result = np.empty(n_events, self.dtype)

        # prepare for an empty event
        empty_events = np.empty(len(anchor_peaks), dtype=self.dtype)
        empty_events["time"] = anchor_peaks["time"]
        empty_events["endtime"] = anchor_peaks["endtime"]

        # build events at once because if a messy environment
        # make the two anchor in the same event
        # it will be considered as event building failure later
        _events = super().compute(_peaks, start, end)
        # touching_windows will return the indices of things repeatedly
        # as long as the things and container are touched so the events here can be repeated
        events = strax.split_touching_windows(_events, anchor_peaks)

        # there should be only one event near the anchor
        l_events = np.array([len(event) for event in events])
        if not np.all(l_events <= 1):
            raise ValueError(f"Expected one event per anchor, got {l_events.max()}!")

        # merge events and placeholders
        _result = np.hstack(
            [events[i] if l_events[i] != 0 else empty_events[i] for i in range(n_events)]
        )

        # sanity checks
        if not np.all(np.diff(_result["time"]) >= 0):
            raise ValueError("Expected the result to be sorted!")
        # some anchors are belonged to a same event
        index = np.unique(_result["time"], return_index=True)[1]
        if not np.all(_result["time"][index][1:] - _result["endtime"][index][:-1] >= 0):
            raise ValueError("Expected the result to be non-overlapping!")
        for n in _result.dtype.names:
            result[n] = _result[n]

        # check if the salting anchor can trigger
        if self.only_salt_s1:
            is_triggering = np.full(len(anchor_peaks), False)
        else:
            is_triggering = self._is_triggering(anchor_peaks)

        # assign the most important parameters
        result["is_triggering"] = is_triggering
        result["salt_number"] = np.unique(peaks_salted["salt_number"])
        result["event_number"] = result["salt_number"]

        if np.any(np.diff(result["time"]) < 0):
            raise ValueError("Expected time to be sorted!")
        return result


class EventBasicsSalted(EventBasics, ExhaustPlugin):
    __version__ = "0.1.0"
    child_plugin = True
    depends_on: Tuple[str, ...] = (
        "events_salted",
        "peaks_salted",
        "peak_proximity_salted",
        "peak_basics",
        "peak_proximity",
        "peak_positions",
    )
    provides = "event_basics_salted"
    data_kind = "events_salted"
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
            self.deps, self.dependencies_by_kind().values(), set.union
        )

        self.peak_properties = tuple(
            list(self.peak_properties) + [("salt_number", np.int64, "Salting number of peaks")]
        )

    def compute(self, events_salted, peaks_salted, peaks):
        if peaks_salted["salt_number"][0] != 0:
            raise ValueError(
                "Expected salt_number to start from 0 because "
                f"{self.__class__.__name__} is a ExhaustPlugin plugin!"
            )

        _peaks = merge_salted_real(peaks_salted, peaks, self._peaks_dtype)

        # we repeated events, so we need to get the unique events
        # because the fully_contained_in function will only return the first container of the thing
        _, index, counts = np.unique(events_salted["time"], return_index=True, return_counts=True)

        _result = np.zeros(len(index), dtype=self.dtype)
        self.set_nan_defaults(_result)

        split_peaks = strax.split_by_containment(_peaks, events_salted[index])

        _result["time"] = events_salted["time"][index]
        _result["endtime"] = events_salted["endtime"][index]

        self.fill_events(_result, events_salted[index], split_peaks)

        for i in [1, 2]:
            if np.all(_result[f"s{i}_salt_number"] < 0):
                warnings.warn(f"Found zero triggered salted S{i}!")

        # recover the original event number
        result = np.repeat(_result, counts)
        result["is_triggering"] = events_salted["is_triggering"]
        result["salt_number"] = events_salted["salt_number"]
        result["event_number"] = events_salted["event_number"]
        return result
