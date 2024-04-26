import warnings
import pytz
import utilix
import strax
from strax import Plugin
import straxen
from straxen import units


export, __all__ = strax.exporter()


@export
class ExhaustPlugin(Plugin):
    """Plugin that exhausts all chunks when fetching data."""

    def _fetch_chunk(self, d, iters, check_end_not_before=None):
        while super()._fetch_chunk(d, iters, check_end_not_before=check_end_not_before):
            pass
        return False

    def do_compute(self, chunk_i=None, **kwargs):
        if chunk_i != 0:
            raise RuntimeError(
                f"{self.__class__.__name__} is an ExhaustPlugin. "
                "It should read all chunks together can process them together."
            )
        return super().do_compute(chunk_i=chunk_i, **kwargs)


@export
class RunMetaPlugin(Plugin):
    """Plugin that provides run metadata."""

    def init_run_meta(self):
        """Get the start and end of the run."""
        if self.real_run_start is None or self.real_run_end is None:
            if self.strict_real_run_time_check:
                raise ValueError("Real run start and end times are not provided!")
            else:
                warnings.warn(
                    "Real run start and end times are not provided. Using utilix to get them."
                )
            if straxen.utilix_is_configured():
                coll = utilix.xent_collection()
            else:
                raise ValueError("Utilix is not configured cannot determine run mode.")
            _doc = coll.find_one(
                {"number": int(self.run_id)}, projection={"start": True, "end": True}
            )
            self.run_start = int(_doc["start"].replace(tzinfo=pytz.utc).timestamp() * units.s)
            self.run_end = int(_doc["end"].replace(tzinfo=pytz.utc).timestamp() * units.s)
        else:
            self.run_start = self.real_run_start
            self.run_end = self.real_run_end
        self.run_time = (self.run_end - self.run_start) / units.s
