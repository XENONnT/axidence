from typing import Tuple
import strax
from strax import Plugin
from straxen import EventShadow, EventAmbience, EventNearestTriggering, EventSEDensity

from ...utils import merge_salted_real


class EventFieldsSalted(Plugin):
    child_plugin = True

    def compute(self, events_salted, peaks_salted, peaks):
        _peaks = merge_salted_real(peaks_salted, peaks, peaks.dtype)
        return super().compute(events_salted, _peaks)


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


class EventNearestTriggeringSalted(EventFieldsSalted, EventNearestTriggering):
    __version__ = "0.0.0"
    depends_on = (
        "event_basics_salted",
        "peaks_salted",
        "peak_nearest_triggering_salted",
        "peak_basics",
        "peak_nearest_triggering",
    )
    provides = "event_nearest_triggering_salted"
    data_kind = "events_salted"
    save_when = strax.SaveWhen.EXPLICIT


class EventSEDensitySalted(EventFieldsSalted, EventSEDensity):
    __version__ = "0.0.0"
    depends_on: Tuple[str, ...] = (
        "event_basics_salted",
        "peaks_salted",
        "peak_se_density_salted",
        "peak_basics",
        "peak_se_density",
    )
    provides = "event_se_density_salted"
    data_kind = "events_salted"
    save_when = strax.SaveWhen.EXPLICIT
