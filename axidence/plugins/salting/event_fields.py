from typing import Tuple
import strax
from straxen import EventShadow, EventAmbience, EventSEDensity

from ...utils import merge_salting_real


class SaltedEventShadow(EventShadow):
    __version__ = "0.0.0"
    depends_on = (
        "event_basics",
        "salting_peaks",
        "salting_peak_shadow",
        "peak_basics",
        "peak_shadow",
    )
    provides = "event_shadow"
    data_kind = "events"
    save_when = strax.SaveWhen.EXPLICIT

    def compute(self, events, salting_peaks, peaks):
        _peaks = merge_salting_real(salting_peaks, peaks, peaks.dtype)
        return super().compute(events, _peaks)


class SaltedEventAmbience(EventAmbience):
    __version__ = "0.0.0"
    depends_on = (
        "event_basics",
        "salting_peaks",
        "salting_peak_ambience",
        "peak_basics",
        "peak_ambience",
    )
    provides = "event_ambience"
    data_kind = "events"
    save_when = strax.SaveWhen.EXPLICIT

    def compute(self, events, salting_peaks, peaks):
        _peaks = merge_salting_real(salting_peaks, peaks, peaks.dtype)
        return super().compute(events, _peaks)


class SaltedEventSEDensity(EventSEDensity):
    __version__ = "0.0.0"
    depends_on: Tuple[str, ...] = (
        "event_basics",
        "salting_peaks",
        "salting_peak_se_density",
        "peak_basics",
        "peak_se_density",
    )
    provides = "event_se_density"
    data_kind = "events"
    save_when = strax.SaveWhen.EXPLICIT

    def compute(self, events, salting_peaks, peaks):
        _peaks = merge_salting_real(salting_peaks, peaks, peaks.dtype)
        return super().compute(events, _peaks)
