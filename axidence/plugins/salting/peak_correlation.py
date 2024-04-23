import strax
from strax import Plugin
from straxen import PeakProximity


class SaltingPeakProximity(PeakProximity):
    __version__ = "0.0.0"
    depends_on = ("salting_peaks", "peak_basics")
    provides = "salting_peak_proximity"
    data_kind = "salting_peaks"
    save_when = strax.SaveWhen.EXPLICIT

    # def infer_dtype(self):
    #     dtype = []


class SaltingPeakShadow(Plugin):
    __version__ = "0.0.0"
    depends_on = ("salting_peaks", "peak_basics", "peak_positions")
    provides = "peak_shadow"
    data_kind = "salting_peaks"
    save_when = strax.SaveWhen.EXPLICIT

    dtype = strax.time_fields


class SaltingPeakAmbience(Plugin):
    __version__ = "0.0.0"
    depends_on = ("salting_peaks", "peak_basics", "peak_positions")
    provides = "peak_ambience"
    data_kind = "salting_peaks"
    save_when = strax.SaveWhen.EXPLICIT

    dtype = strax.time_fields


class SaltingPeakSEDensity(Plugin):
    __version__ = "0.0.0"
    depends_on = ("salting_peaks", "peak_basics", "peak_positions")
    provides = "salting_peak_se_density"
    data_kind = "salting_peaks"
    save_when = strax.SaveWhen.EXPLICIT

    dtype = strax.time_fields
