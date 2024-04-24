import strax
from strax import Plugin


class SaltedEventShadow(Plugin):
    __version__ = "0.0.0"
    depends_on = ("event_basics", "salting_peaks", "peak_shadow")
    provides = "event_shadow"
    data_kind = "events"
    save_when = strax.SaveWhen.EXPLICIT

    dtype = strax.time_fields


class SaltedEventAmbience(Plugin):
    __version__ = "0.0.0"
    depends_on = ("event_basics", "salting_peaks", "peak_ambience")
    provides = "event_ambience"
    data_kind = "events"
    save_when = strax.SaveWhen.EXPLICIT

    dtype = strax.time_fields


class SaltedEventSEDensity(Plugin):
    __version__ = "0.0.0"
    depends_on = (
        "event_basics",
        "salting_peaks",
        "salting_peak_se_density",
        "peak_basics",
        "peak_se_density",
    )
    provides = "event_se_density"
    data_kind = "events"
    save_when = strax.SaveWhen.EXPLICIT

    dtype = strax.time_fields
