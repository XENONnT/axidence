import warnings
from typing import Tuple
import numpy as np
import utilix
import strax
import straxen
from straxen import units, EventBasics, EventPositions
from straxen.misc import kind_colors

from ...utils import copy_dtype


kind_colors.update(
    {
        "salting_events": "#0080ff",
        "salting_peaks": "#00ffff",
    }
)


class SaltingEvents(EventPositions, EventBasics):
    __version__ = "0.0.0"
    depends_on: Tuple = tuple()
    provides = "salting_events"
    data_kind = "salting_events"
    save_when = strax.SaveWhen.EXPLICIT

    salting_seed = straxen.URLConfig(
        default=None,
        type=(int, None),
        help="Seed for salting",
    )

    real_run_start = straxen.URLConfig(
        default=None,
        type=(int, None),
        help="Real start time of run [ns]",
    )

    real_run_end = straxen.URLConfig(
        default=None,
        type=(int, None),
        help="Real start time of run [ns]",
    )

    strict_real_run_time_check = straxen.URLConfig(
        default=True,
        type=bool,
        help="Whether to strictly check the real run time is provided",
    )

    salting_rate = straxen.URLConfig(
        default=10,
        type=(int, float),
        help="Rate of salting in Hz",
    )

    s1_area_range = straxen.URLConfig(
        default=(1, 150),
        type=(list, tuple),
        help="Range of S1 area",
    )

    s2_area_range = straxen.URLConfig(
        default=(1e2, 2e4),
        type=(list, tuple),
        help="Range of S2 area",
    )

    s1_distribution = straxen.URLConfig(
        default="",
        type=str,
        help="S1 distribution shape",
    )

    s2_distribution = straxen.URLConfig(
        default="",
        type=str,
        help="S2 distribution shape",
    )

    veto_length_run_start = straxen.URLConfig(
        default=10**9,
        type=int,
        track=True,
        help="Min time delay in [ns] for events towards the run start boundary",
    )

    veto_length_run_end = straxen.URLConfig(
        default=10**8,
        type=int,
        track=True,
        help="Min time delay in [ns] for events towards the run end boundary",
    )

    s1_n_hits_tight_coincidence = straxen.URLConfig(
        default=2,
        type=int,
        track=True,
        help="Will assign the events's ``s1_n_hits`` and ``s1_tight_coincidence`` by this",
    )

    n_drift_time_window = straxen.URLConfig(
        default=5,
        type=int,
        track=True,
        help="How many max drift time will the event builder extend",
    )

    def _set_posrec_save(self):
        self.posrec_save = []
        self.pos_rec_labels = []

    def refer_dtype(self):
        return strax.merged_dtype(
            [
                strax.to_numpy_dtype(super(EventPositions, self).infer_dtype()),
                strax.to_numpy_dtype(super(SaltingEvents, self).infer_dtype()),
            ]
        )

    def infer_dtype(self):
        dtype_reference = self.refer_dtype()
        required_names = ["time", "endtime", "s1_center_time", "s2_center_time"]
        required_names += ["s1_area", "s2_area", "s1_n_hits", "s1_tight_coincidence"]
        required_names += ["x", "y", "z", "drift_time", "s2_x", "s2_y", "z_naive"]
        dtype = copy_dtype(dtype_reference, required_names)
        # since event_number is int64 in event_basics
        dtype += [(("Salting number of events", "salt_number"), np.int64)]
        return dtype

    def init_rng(self):
        if self.salting_seed is None:
            self.rng = np.random.default_rng(seed=int(self.run_id))
        else:
            self.rng = np.random.default_rng(seed=self.salting_seed)

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
            self.run_start = int(_doc["start"].timestamp() * units.s)
            self.run_end = int(_doc["end"].timestamp() * units.s)
        else:
            self.run_start = self.real_run_start
            self.run_end = self.real_run_end

    def sample_time(self):
        """Sample the time according to the start and end of the run."""
        self.event_time_interval = units.s // self.salting_rate

        if units.s / self.salting_rate < self.drift_time_max * self.n_drift_time_window * 2:
            raise ValueError("Salting rate is too high according the drift time window!")

        time = np.arange(
            self.run_start + self.veto_length_run_start,
            self.run_end - self.veto_length_run_end,
            self.event_time_interval,
        ).astype(np.int64)
        self.time_left = self.event_time_interval // 2
        self.time_right = self.event_time_interval - self.time_left
        return time

    def inverse_field_distortion(self, x, y, z):
        # TODO: implement detailed inverse field distortion
        return x, y, z

    def set_chunk_splitting(self):
        """Split the rujn into chunks to prevent oversize chunk."""
        self.max_n_events_in_chunk = round(
            self.chunk_target_size_mb * 1e6 / self.dtype.itemsize * 0.9
        )
        slices_idx = np.unique(
            np.append(
                np.arange(self.n_events, dtype=int)[:: self.max_n_events_in_chunk], self.n_events
            )
        )
        self.slices = np.vstack([slices_idx[:-1], slices_idx[1:]]).T.astype(int).tolist()

        self.time_left = self.event_time_interval // 2
        self.time_right = self.event_time_interval - self.time_left

    def setup(self):
        """Sample the features of events."""
        super(EventPositions, self).setup()
        super(SaltingEvents, self).setup()

        self.init_rng()
        self.init_run_meta()

        time = self.sample_time()
        self.n_events = len(time)
        self.salting_events = np.empty(self.n_events, dtype=self.dtype)
        self.salting_events["salt_number"] = np.arange(self.n_events)
        self.salting_events["time"] = time
        self.salting_events["endtime"] = time

        self.salting_events["s1_n_hits"] = self.s1_n_hits_tight_coincidence
        self.salting_events["s1_tight_coincidence"] = self.s1_n_hits_tight_coincidence

        theta = self.rng.random(size=self.n_events) * 2 * np.pi - np.pi
        r = np.sqrt(self.rng.random(size=self.n_events)) * straxen.tpc_r
        self.salting_events["x"] = np.cos(theta) * r
        self.salting_events["y"] = np.sin(theta) * r
        self.salting_events["z"] = -self.rng.random(size=self.n_events) * straxen.tpc_z
        s2_x, s2_y, z_naive = self.inverse_field_distortion(
            self.salting_events["x"],
            self.salting_events["y"],
            self.salting_events["z"],
        )
        self.salting_events["s2_x"] = s2_x
        self.salting_events["s2_y"] = s2_y
        self.salting_events["z_naive"] = z_naive
        self.salting_events["drift_time"] = (
            self.electron_drift_velocity * self.electron_drift_time_gate
            - self.salting_events["z_naive"]
        ) / self.electron_drift_velocity

        self.salting_events["s1_center_time"] = time - self.salting_events["drift_time"]
        self.salting_events["s2_center_time"] = time

        s1_area_range = (float(self.s1_area_range[0]), float(self.s1_area_range[1]))
        s2_area_range = (float(self.s2_area_range[0]), float(self.s2_area_range[1]))
        self.salting_events["s1_area"] = np.exp(
            self.rng.uniform(np.log(s1_area_range[0]), np.log(s1_area_range[1]), size=self.n_events)
        )
        self.salting_events["s1_area"] = np.clip(self.salting_events["s1_area"], *s1_area_range)
        self.salting_events["s2_area"] = np.exp(
            self.rng.uniform(np.log(s2_area_range[0]), np.log(s2_area_range[1]), size=self.n_events)
        )
        self.salting_events["s2_area"] = np.clip(self.salting_events["s2_area"], *s2_area_range)

        self.set_chunk_splitting()

    def compute(self, chunk_i):
        """Copy and assign the salting events into chunk."""
        indices = self.slices[chunk_i]

        if chunk_i == 0:
            start = self.run_start
        else:
            start = self.salting_events["time"][indices[0]] - self.time_left

        if chunk_i == len(self.slices) - 1:
            end = self.run_end
        else:
            end = self.salting_events["time"][indices[1] - 1] + self.time_right
        return self.chunk(start=start, end=end, data=self.salting_events[indices[0] : indices[1]])

    def is_ready(self, chunk_i):
        if chunk_i < len(self.slices):
            return True
        else:
            return False

    def source_finished(self):
        return True
