import numpy as np
import strax
from strax import Plugin
from straxen import EventShadow, EventAmbience

from ...utils import merge_salted_real


class EventFieldsSalted(Plugin):
    __version__ = "0.1.0"
    child_plugin = True

    def compute(self, events_salted, peaks_salted, peaks):
        _peaks = merge_salted_real(peaks_salted, peaks, peaks.dtype)
        _, index, counts = np.unique(events_salted["time"], return_index=True, return_counts=True)
        _result = super().compute(events_salted[index], _peaks)
        result = np.repeat(_result, counts)
        return result


class EventShadowSalted(EventFieldsSalted, EventShadow):
    __version__ = "0.0.0"
    depends_on = (
        "event_basics_salted",
        "peaks_salted",
        "peak_shadow_salted",
        "peak_basics",
        "peak_shadow",
    )
    provides = "event_shadow_salted"
    data_kind = "events_salted"
    save_when = strax.SaveWhen.EXPLICIT


class EventAmbienceSalted(EventFieldsSalted, EventAmbience):
    __version__ = "0.0.0"
    depends_on = (
        "event_basics_salted",
        "peaks_salted",
        "peak_ambience_salted",
        "peak_basics",
        "peak_ambience",
    )
    provides = "event_ambience_salted"
    data_kind = "events_salted"
    save_when = strax.SaveWhen.EXPLICIT


# SR0 release: EventNearestTriggeringSalted and EventSEScoreSalted are
# dropped because the underlying straxen.EventNearestTriggering /
# EventSEScore plugins don't exist in straxen 1.7.x.
